import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simple fluid + queue simulator
# -----------------------------
# Units:
# - time: seconds
# - link capacity: packets/sec
# - cwnd: packets
# - RTT: seconds
# - queue: packets

def jains_fairness(x):
    x = np.array(x, dtype=float)
    if np.all(x == 0):
        return 0.0
    return (x.sum() ** 2) / (len(x) * (x**2).sum())

class Flow:
    def __init__(self, algo, base_rtt, init_cwnd=10.0):
        self.algo = algo
        self.base_rtt = base_rtt
        self.cwnd = init_cwnd
        self.ssthresh = 1e9  # ignore slow-start for simplicity
        self.alpha = 0.0     # for DCTCP-like
        self.last_q = 0.0    # for APCC
        self.last_dq = 0.0   # for APCC
        self.sent = 0.0
        self.acked = 0.0
        self.lost = 0.0

    def sending_rate(self, rtt):
        # rate ~= cwnd / rtt (packets/sec)
        return max(self.cwnd / max(rtt, 1e-6), 0.0)

def simulate(
    algo_name,
    n_flows=8,
    sim_time=2.0,
    dt=0.001,
    base_rtt=0.0002,        # 200 microseconds (data center-ish)
    capacity_pps=1e6,       # packets/sec (just a scale)
    buffer_pkts=2000,       # queue buffer size in packets
    ecn_k=400,              # ECN marking threshold (packets)
    ecn_maxp=1.0,           # max mark probability
    pkt_size_bytes=1500
):
    steps = int(sim_time / dt)
    flows = [Flow(algo_name, base_rtt) for _ in range(n_flows)]
    q = 0.0  # queue length in packets

    # Logs
    t_log = np.zeros(steps)
    q_log = np.zeros(steps)
    delay_log = np.zeros(steps)
    thr_log = np.zeros(steps)
    loss_log = np.zeros(steps)
    cwnd_log = np.zeros((n_flows, steps))

    bytes_per_pkt = pkt_size_bytes

    for i in range(steps):
        t = i * dt

        # current RTT includes queueing delay: q / capacity
        q_delay = q / capacity_pps
        rtt = base_rtt + q_delay

        # Total offered load (packets/sec)
        rates = np.array([f.sending_rate(rtt) for f in flows])
        offered_pps = rates.sum()

        # Service this timestep
        served = min(q + offered_pps * dt, capacity_pps * dt)  # packets served
        arrivals = offered_pps * dt

        # Queue update before drops
        q_next = q + arrivals - served

        # Drops if buffer overflow
        drops = max(q_next - buffer_pkts, 0.0)
        if drops > 0:
            q_next = buffer_pkts

        # Estimate per-flow delivered and dropped proportionally to send rate
        if offered_pps > 1e-12:
            share = rates / offered_pps
        else:
            share = np.zeros_like(rates)

        delivered_pkts = (served * share)  # goodput approximation
        dropped_pkts = (drops * share)

        # ECN marking probability based on queue length
        # simple RED-like: 0 below K, ramps to max at buffer
        if q_next <= ecn_k:
            mark_p = 0.0
        else:
            mark_p = min(ecn_maxp * (q_next - ecn_k) / max(buffer_pkts - ecn_k, 1e-9), ecn_maxp)

        # Update per-flow counters
        for idx, f in enumerate(flows):
            f.sent += rates[idx] * dt
            f.acked += delivered_pkts[idx]
            f.lost += dropped_pkts[idx]

        # Congestion control updates (once per RTT-ish is complex;
        # here we do a small update every dt, scaled by dt/rtt)
        gain = dt / max(rtt, 1e-6)

        for f in flows:
            if algo_name == "reno":
                # AIMD: add ~1 packet per RTT, cut in half on loss
                if drops > 0:
                    f.cwnd = max(1.0, f.cwnd * 0.5)
                else:
                    f.cwnd += 1.0 * gain

            elif algo_name == "dctcp":
                # DCTCP-like:
                # estimate fraction marked and reduce cwnd proportionally
                # alpha <- (1-g)*alpha + g*F, where F ~ mark_p
                g = 1/16  # typical EWMA gain
                F = mark_p
                f.alpha = (1 - g) * f.alpha + g * F
                if drops > 0:
                    # loss still triggers stronger cut
                    f.cwnd = max(1.0, f.cwnd * 0.5)
                else:
                    # proportional reduction: cwnd <- cwnd*(1 - alpha/2) once per RTT; approximate continuously
                    f.cwnd = max(1.0, f.cwnd * (1 - 0.5 * f.alpha * gain) + 1.0 * gain)

            elif algo_name == "apcc":
                # APCC (proposal): predict queue trend and adapt AI/MD parameters.
                # - Predict next queue using last delta
                dq = q_next - f.last_q
                pred_q = q_next + dq

                # Confidence heuristic: if queue trend is stable, allow more aggressive probing
                # else fall back closer to AIMD.
                stability = np.exp(-abs(dq) / 50.0)  # 0..1

                # Adaptive additive increase (AI)
                # more stable + low queue -> higher increase; high queue -> lower increase
                ai_base = 1.0
                ai = ai_base * (0.3 + 0.7 * stability) * (1.0 - min(pred_q / buffer_pkts, 1.0))

                # Adaptive multiplicative decrease (MD)
                # stronger decrease when predicted queue is high or marking high
                md_base = 0.5
                md = md_base + 0.4 * min(pred_q / buffer_pkts, 1.0) + 0.3 * mark_p  # up to ~1.2

                if drops > 0:
                    f.cwnd = max(1.0, f.cwnd * 0.5)  # hard loss fallback
                else:
                    # if congestion predicted (ECN or high queue), decrease; else increase
                    if (mark_p > 0.0) or (pred_q > ecn_k):
                        # proportional-ish decrease
                        f.cwnd = max(1.0, f.cwnd * (1 - md * gain))
                    else:
                        f.cwnd += ai * gain

                f.last_q = q_next
                f.last_dq = dq

            else:
                raise ValueError("Unknown algo")

        # Log
        t_log[i] = t
        q_log[i] = q_next
        delay_log[i] = q_next / capacity_pps  # queueing delay
        # throughput in Gbps
        thr_bytes_per_s = (delivered_pkts.sum() / dt) * bytes_per_pkt
        thr_gbps = (thr_bytes_per_s * 8) / 1e9
        thr_log[i] = thr_gbps
        loss_log[i] = drops

        for k, f in enumerate(flows):
            cwnd_log[k, i] = f.cwnd

        q = q_next

    # Summary metrics
    total_acked = np.array([f.acked for f in flows])
    total_sent = np.array([f.sent for f in flows])
    total_lost = np.array([f.lost for f in flows])

    # Average per-flow throughput (Gbps)
    per_flow_thr_gbps = (total_acked * bytes_per_pkt * 8) / (sim_time * 1e9)

    avg_delay_us = delay_log.mean() * 1e6
    avg_thr_gbps = per_flow_thr_gbps.sum()
    loss_rate = (total_lost.sum() / max(total_sent.sum(), 1e-9))
    fairness = jains_fairness(per_flow_thr_gbps)

    results = {
        "t": t_log,
        "q": q_log,
        "q_delay": delay_log,
        "thr_gbps": thr_log,
        "cwnd": cwnd_log,
        "avg_delay_us": avg_delay_us,
        "avg_thr_gbps": avg_thr_gbps,
        "loss_rate": loss_rate,
        "fairness": fairness,
        "per_flow_thr_gbps": per_flow_thr_gbps,
    }
    return results

def main():
    # Use same network for all algos
    algos = ["reno", "dctcp", "apcc"]
    runs = {a: simulate(a) for a in algos}

    # Print summary
    print("\n=== Summary (same topology/params across algorithms) ===")
    print(f"{'Algo':<8} {'AvgThr(Gbps)':>12} {'AvgQDelay(us)':>14} {'LossRate':>10} {'JainFair':>10}")
    for a in algos:
        r = runs[a]
        print(f"{a:<8} {r['avg_thr_gbps']:>12.3f} {r['avg_delay_us']:>14.2f} {r['loss_rate']:>10.4f} {r['fairness']:>10.3f}")

    # Plot 1: Queue length
    plt.figure()
    for a in algos:
        plt.plot(
            runs[a]["t"],
            runs[a]["q"],
            label=a.upper(),
            linewidth=1.5
        )
    plt.title("Queue Length vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Queue (packets)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 2: Queueing delay (us)
    plt.figure()
    for a in algos:
        plt.plot(
            runs[a]["t"],
            runs[a]["q_delay"] * 1e6,
            label=a.upper(),
            linewidth=1.5
        )
    plt.title("Queueing Delay vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Queueing Delay (µs)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 3: Total throughput (Gbps)
    plt.figure()
    for a in algos:
        plt.plot(
            runs[a]["t"],
            runs[a]["thr_gbps"],
            label=a.upper(),
            linewidth=1.5
        )
    plt.title("Aggregate Goodput vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Goodput (Gbps)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Plot 4: Average cwnd
    plt.figure()
    for a in algos:
        cw = runs[a]["cwnd"]
        plt.plot(
            runs[a]["t"],
            cw.mean(axis=0),
            label=a.upper(),
            linewidth=1.5
        )
    plt.title("Average Congestion Window vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Average CWND (packets)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
