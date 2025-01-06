import csv

result = {}

with open("./benchmarks/tracking_cif/rising/mg/rising_mg_0x17.csv") as file:
    result["0x17"] = []
    reader = csv.DictReader(file)
    for row in reader:
        result["0x17"].append(float(row["job_elapsed(seconds)"]) * 1000)


with open("./benchmarks/tracking_cif/rising/mg/rising_mg_0x18.csv") as file:
    result["0x18"] = []
    reader = csv.DictReader(file)
    for row in reader:
        result["0x18"].append(float(row["job_elapsed(seconds)"]) * 1000)


with open("./benchmarks/tracking_cif/rising/mg/rising_mg_0x19.csv") as file:
    result["0x19"] = []
    reader = csv.DictReader(file)
    for row in reader:
        result["0x19"].append(float(row["job_elapsed(seconds)"]) * 1000)


with open("./benchmarks/tracking_cif/rising/rising_jailhouse_enabled.csv") as file:
    result["mempol jailhouse enabled"] = []
    reader = csv.DictReader(file)
    for row in reader:
        result["mempol jailhouse enabled"].append(float(row["job_elapsed(seconds)"]) * 1000)


with open("./benchmarks/tracking_cif/rising/rising_jailhouse_disabled.csv") as file:
    result["mempol jailhouse disabled"] = []
    reader = csv.DictReader(file)
    for row in reader:
        result["mempol jailhouse disabled"].append(float(row["job_elapsed(seconds)"]) * 1000)

with open("./benchmarks_L2_no_interference/tracking_cif/rising/rising.csv") as file:
    result["mempol old"] = []
    reader = csv.DictReader(file)
    for row in reader:
        result["mempol old"].append(float(row["job_elapsed(seconds)"]) * 1000)

# plot results
import matplotlib.pyplot as plt

plt.plot(range(100, 4001, 100), result["0x17"], label="0x17")
plt.plot(range(100, 4001, 100), result["0x18"], label="0x18")
plt.plot(range(100, 4001, 100), result["0x19"], label="0x19")
plt.plot(range(100, 4001, 100), result["mempol jailhouse enabled"], label="mempol jailhouse enabled")
plt.plot(range(100, 4001, 100), result["mempol jailhouse disabled"], label="mempol jailhouse disabled")
plt.plot(range(100, 4001, 100), result["mempol old"][:40], label="mempol previous")
plt.xlabel("Budget (MB/s)")
plt.ylabel("Execution Time (ms)")
plt.title("Execution Time vs Budget")
plt.legend()
plt.show()
