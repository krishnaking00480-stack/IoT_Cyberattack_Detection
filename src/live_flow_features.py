import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "reports" / "live_packets.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "live_flow_features.csv"

# Load captured packet metadata
df = pd.read_csv(CSV_PATH)

# Convert numeric fields safely
numeric_columns = [
    "frame.time_epoch",
    "frame.len",
    "ip.proto",
    "tcp.srcport",
    "tcp.dstport",
    "udp.srcport",
    "udp.dstport",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Keep only IPv4 packets with source and destination IPs
df = df.dropna(subset=["ip.src", "ip.dst", "frame.time_epoch", "frame.len"])

# Use TCP or UDP ports when available
df["src_port"] = df["tcp.srcport"].fillna(df["udp.srcport"]).fillna(0)
df["dst_port"] = df["tcp.dstport"].fillna(df["udp.dstport"]).fillna(0)

# Group packets into directional flows
flow_columns = [
    "ip.src",
    "ip.dst",
    "src_port",
    "dst_port",
    "ip.proto",
]

flows = []

for flow_key, group in df.groupby(flow_columns, dropna=False):
    start_time = group["frame.time_epoch"].min()
    end_time = group["frame.time_epoch"].max()
    duration = max(end_time - start_time, 0.000001)

    packet_count = len(group)
    total_bytes = group["frame.len"].sum()

    flows.append({
        "src_ip": flow_key[0],
        "dst_ip": flow_key[1],
        "src_port": flow_key[2],
        "dst_port": flow_key[3],
        "protocol": flow_key[4],
        "packet_count": packet_count,
        "total_bytes": total_bytes,
        "duration_seconds": duration,
        "average_packet_size": group["frame.len"].mean(),
        "packet_rate": packet_count / duration,
        "byte_rate": total_bytes / duration,
    })

# Save the flow summary
flow_df = pd.DataFrame(flows)
flow_df.to_csv(OUTPUT_PATH, index=False)

print("Flow feature extraction completed.")
print(f"Flows created: {len(flow_df)}")
print(f"Saved to: {OUTPUT_PATH}")
print(flow_df.head())
