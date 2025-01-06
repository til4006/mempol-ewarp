import matplotlib.pyplot as plt
import csv
import math
import re
import json
import os
from argparse import ArgumentParser
import pickle
import random


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument(
        "-l",
        "--load-cache",
        action="store_true",
        help="Flag specifies, if an already parsed json file should be loaded as input, if it exists.",
    )
    parser.add_argument(
        "-s",
        "--store-cache",
        action="store_true",
        help="Flag specifies, if the parsed data should be stored to speedup future execution.",
    )
    args = parser.parse_args()
    # args.benchmark = "benchmarks/" + args.benchmark
    return args


class BenchmarkParser(object):
    TRACE_CSV_BASENAME = "trace.csv"
    TRACE_INFO_BASENAME = "trace.info"
    TIMING_CSV_BASENAME = "timing.csv"
    CACHED_DATA_BASENAME = "cached_data.pkl"
    RISING_BUDGET_BASENAME = "rising/rising.csv"

    def __init__(self, args):
        self.load_cache = args.load_cache
        self.store_cache = args.store_cache
        self.benchmark = args.benchmark
        self.e = ""
        self.create_filenames()

    def create_filenames(self):
        self.benchmark_dir = os.path.join("benchmarks", self.benchmark)
        self.trace_csv_path = os.path.join(self.benchmark_dir, self.TRACE_CSV_BASENAME)
        self.trace_info_path = os.path.join(
            self.benchmark_dir, self.TRACE_INFO_BASENAME
        )
        self.timing_info_path = os.path.join(
            self.benchmark_dir, self.TIMING_CSV_BASENAME
        )
        self.cached_data_path = os.path.join(
            self.benchmark_dir, self.CACHED_DATA_BASENAME
        )
        self.rising_data_path = os.path.join(
            self.benchmark_dir, self.RISING_BUDGET_BASENAME
        )
        return

    def check_benchmark(self):
        result = True
        if not os.path.isdir(self.benchmark_dir):
            self.e = f"Missing benchmark folder: {self.benchmark_dir}"
            result = False
        elif not os.path.isfile(self.trace_csv_path):
            self.e = f"Missing trace csv file: {self.trace_csv_path}"
            result = False
        elif not os.path.isfile(self.trace_info_path):
            self.e = f"Missing trace info file: {self.trace_info_path}"
            result = False
        elif not os.path.isfile(self.timing_info_path):
            self.e = f"Missing timing csv file: {self.timing_info_path}"
            result = False
        elif not os.path.isfile(self.rising_data_path):
            self.e = f"Missing rising budget csv file: {self.rising_data_path}"
            result = False
        else:
            result = True
        return result

    def error_handling(self):
        print(self.e)
        return

    def parse(self):
        result = None
        if not self.check_benchmark():
            self.error_handling()
            result = None
        else:
            if self.load_cache and self.has_cached_data():
                result = self.parse_benchmark_data_cached()
            else:
                result = self.parse_benchmark_data()
        return result

    def has_cached_data(self):
        result = True
        if not os.path.isfile(self.cached_data_path):
            result = False
        return result

    def load_cached_data(self):
        with open(self.cached_data_path, "rb") as file:
            data = pickle.load(file)
        return data

    def store_cached_data(self, data):
        with open(self.cached_data_path, "wb") as file:
            pickle.dump(data.jobs, file)

    def parse_benchmark_data_cached(self):
        benchmark = Benchmark(self.benchmark)
        benchmark.jobs = self.load_cached_data()
        benchmark.measured_data = self.parse_rising_data()
        benchmark.measured_data_with_inf = self.parse_rising_data_with_inf()
        return benchmark

    def parse_benchmark_data(self):
        benchmark = Benchmark(self.benchmark)
        self.parse_job_data()
        self.parse_trace_info()
        self.parse_trace_data()
        benchmark.cores = self.cores
        benchmark.jobs = self.jobs
        benchmark.measured_data = self.parse_rising_data()
        benchmark.measured_data_with_inf = self.parse_rising_data_with_inf()
        if self.store_cache:
            self.store_cached_data(benchmark)
        return benchmark

    def parse_rising_data(self, measured_budget=range(100, 4001, 100)):
        result = {}
        with open(self.rising_data_path, "r") as file:
            reader = csv.DictReader(file)
            result["measured_time_ms"] = []
            for row in reader:
                result["measured_time_ms"].append(
                    float(row["job_elapsed(seconds)"]) * 1000
                )
            result["measured_time_ms"] = result["measured_time_ms"][:len(measured_budget)]
            result["measured_budget"] = measured_budget
        return result
    
    def parse_rising_data_with_inf(self, measured_budget=range(100, 4001, 100)):
        result = {}
        for inf in range(500, 4001, 500):
            result[inf] = {}
            if not os.path.exists(self.rising_data_path.replace("rising.csv", f"inf_{inf:08d}")):
                print(f"Missing rising data for inf {inf}")
                return None
            with open(self.rising_data_path.replace("rising.csv", f"inf_{inf:08d}/rising.csv"), "r") as file:
                reader = csv.DictReader(file)
                result[inf]["measured_time_ms"] = []
                for row in reader:
                    result[inf]["measured_time_ms"].append(
                        float(row["job_elapsed(seconds)"]) * 1000
                    )
                result[inf]["measured_budget"] = measured_budget
        return result

    def parse_job_data(self, coreid=0):
        self.jobs = []
        with open(self.timing_info_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                self.jobs.append(JobData(TimingInfo(row), coreid=coreid))
        return self.jobs

    def parse_trace_info(self):
        self.trace_infos = {}
        with open(self.trace_info_path, "r") as file:
            regex = TraceInfo.create_regex()
            for line in file.readlines():
                trace_info = TraceInfo(regex.search(line).groupdict())
                self.trace_infos[trace_info.block] = trace_info
        return self.trace_infos

    def parse_trace_data(self, num_cores=4):
        self.cores = [
            CoreData(coreid, self.trace_infos, self.jobs)
            for coreid in range(0, num_cores)
        ]
        with open(self.trace_csv_path, "r") as file:
            reader = TraceData.create_csv_dict_reader(file, num_cores)
            current_jobid = 0
            for row in reader:
                for core in self.cores:
                    current_jobid = core.add_trace_data(row, current_jobid)
        return self.cores


class Benchmark(object):
    def __init__(self, name):
        self.benchmark_dir = os.path.join("benchmarks", name)
        self.name = name
        self.cores = []
        self.jobs = []
        self.measured_data = {}
        self.measured_data_with_inf = {}


class CoreData(object):
    def __init__(self, coreid, trace_infos, jobs):
        self.trace_infos = trace_infos
        self.jobs = jobs
        self.coreid = coreid
        self.trace_data = []

    def add_trace_data(self, data, current_jobid):
        trace_data = TraceData(data, self.coreid)
        trace_data.map_trace_info(self.trace_infos)
        current_jobid = trace_data.map_job(self.jobs, current_jobid, self.coreid)
        self.trace_data.append(trace_data)
        return current_jobid


class TraceData(object):
    def create_fieldnames(num_cores):
        fieldnames = ["loop"] + [f"core{i}_regulated" for i in range(0, num_cores)]
        for i in range(0, num_cores):
            fieldnames.append(f"core{i}_read")
            fieldnames.append(f"core{i}_write")
        return fieldnames

    def create_csv_dict_reader(file, num_cores):
        return csv.DictReader(
            file, delimiter=";", fieldnames=TraceData.create_fieldnames(num_cores)
        )

    def __init__(self, data, coreid):
        self.loop = int(data["loop"])
        self.regulated = int(data[f"core{coreid}_regulated"])
        self.read = int(data[f"core{coreid}_read"])
        self.write = int(data[f"core{coreid}_write"])
        self.trace_info = None
        self.job = None

    def map_trace_info(self, trace_infos, num_blocks=256):
        id = math.floor(self.loop / num_blocks)
        self.trace_info = trace_infos.get(id)

    def map_job(self, jobs, jobid, coreid):
        if jobid < len(jobs):
            current_job = jobs[jobid]
            if (
                current_job.start_time()
                <= self.get_monotonic()
                <= current_job.end_time()
            ):
                self.do_job_mapping(current_job, coreid)
            elif self.get_monotonic() > current_job.start_time():
                jobid += 1
                self.map_job(jobs, jobid, coreid)
        return jobid

    def do_job_mapping(self, job, coreid):
        if job.coreid == coreid:
            self.job = job
            job.trace_data.append(self)

    def get_monotonic(self, num_blocks=256, sample_time=0.00001):
        if self.trace_info == None:
            return -1
        offset = self.loop % num_blocks
        return self.trace_info.monotonic + offset * sample_time

    def get_weighted_access(self, weight_factor=[1, 1]):
        return self.read * weight_factor[0] + self.write * weight_factor[1]


class JobData(object):
    def __init__(self, timing_info=None, coreid=0):
        self.coreid = coreid
        self.timing_info = timing_info
        self.trace_data = []

    def start_time(self):
        return self.timing_info.period_start

    def end_time(self):
        return self.timing_info.job_end


class TimingInfo(object):
    def __init__(self, data):
        self.period_start = float(data["period_start(seconds)"])
        self.job_end = float(data["job_end(seconds)"])


class TraceInfo(object):
    def create_regex():
        return re.compile(r"# block:(?P<block>\d+) .*monotonic:(?P<monotonic>\d+.\d+)")

    def __init__(self, data):
        self.block = int(data["block"])
        self.monotonic = float(data["monotonic"])


def get_envelope(
    task, processing_element, runs, delta=10000, weight_factor=[1000, 1000]
):
    L_i = 0
    M_j = [processing_element]
    sigma_j = []
    for r in runs:
        x_r = 0
        L_r = 0
        number_of_samples = len(r.trace_data)
        for h in range(0, number_of_samples):
            L_r += 1
            x_r += r.trace_data[h].get_weighted_access(weight_factor)
            if L_r > L_i:
                L_i = L_r
                sigma_j.append(
                    {
                        "x+j": max(sigma_j[h - 1]["x+j"] if h > 0 else 0, x_r),
                        "x-j": x_r,
                    }
                )
                M_j.append(sigma_j[h])
            else:
                sigma_j[h]["x+j"] = (
                    x_r if x_r > sigma_j[h]["x+j"] else sigma_j[h]["x+j"]
                )
                sigma_j[h]["x-j"] = (
                    x_r if x_r < sigma_j[h]["x-j"] else sigma_j[h]["x-j"]
                )
    return M_j, L_i * delta


def get_wcet_cpu(
    task,
    M_j,
    CPU_k,
    t_ovh=0,
    x_ovh=0,
    t_stall=0,
    P=10000 * 10,
    delta=10000,
    Q=10000 * 10,
):
    t_add = P
    x_off = 0
    t_s = 0
    x_s = 0
    h = 0
    for h, sigma_j in enumerate(M_j[1:]):
        x_plus = sigma_j["x+j"]
        x_minus = sigma_j["x-j"]
        t = delta * h
        if t - t_s >= P:
            t_add = t_add + t_stall * x_s + t_ovh
            t_s = t - ((t - t_s) - P)
            x_s = min(x_plus, x_minus + x_off)
        if x_plus - x_s >= Q:
            t_add += P - (t - t_s) + t_ovh
            t_s = t
            x_off = min(x_plus, x_off + Q)
            x_s = min(x_plus, x_minus + x_off)
    return t + t_add


def get_wcet_cpu_mempol_sw(
    task,
    M_j,
    CPU_k,
    t_ovh=0,
    x_ovh=0,
    t_stall=0,
    P=10000 * 10,
    delta=10000,
    Q=10000 * 10,
    window_size=5,
):
    t_add = 0
    x_off = 0
    t_s = 0
    x_s = 0
    h = 0
    history = []
    for h, sigma_j in enumerate(M_j[1:]):
        x_plus = sigma_j["x+j"]
        x_minus = sigma_j["x-j"]
        t = delta * h
        history.append(min(x_plus, max(x_minus, x_off)))
        if len(history) > window_size:
            x_s = history.pop(0)
        else:
            x_s = x_s
        if x_plus - x_s >= Q * len(history):
            overshoot = x_plus - x_s - Q * len(history)
            t_add += math.ceil((overshoot) / Q) * delta
            x_s = x_plus
            x_off = x_plus + Q * len(history)  # max(x_off, x_plus) + Q * len(history)
            history = []
    return t + t_add


def get_wcet_cpu_mempol_tb(
    task,
    M_j,
    CPU_k,
    t_ovh=0,
    x_ovh=0,
    t_stall=0,
    P=10000 * 10,
    delta=10000,
    Q=10000 * 10, # token size
    window_size=5,
):
    t_add = 0
    x_off = 0
    x_s = 0
    h = 0
    token_bucket = 0
    for h, sigma_j in enumerate(M_j[1:]):
        x_plus = sigma_j["x+j"]
        x_minus = sigma_j["x-j"]
        t = delta * h
        token_bucket = min(Q * window_size, token_bucket + Q)
        token_bucket -= x_plus - x_s
        if token_bucket <= 0:
            t_add += math.ceil(abs(token_bucket) / Q) * delta
            token_bucket += math.ceil(abs(token_bucket) / Q) * Q
            x_off = x_plus
        x_s = max(x_minus, x_off)
    return t + t_add


def plot_run(
    runs, index, filename, sample_rate_ns=10_000, weight_factor=[1000, 1000], title=""
):
    run = runs[index]
    accesses = [td.get_weighted_access(weight_factor) for td in run.trace_data]
    t = [i * sample_rate_ns / 1_000_000 for i in range(len(run.trace_data))]
    plt.plot(t, accesses)
    plt.xlabel("Time (ms)")
    plt.ylabel("Weighted Accesses")
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def plot_envelope(
    envelope, filename, colors=["b", "c"], sample_rate_ns=10_000, title=""
):
    t = [i * sample_rate_ns / 1_000_000 for i in range(len(envelope[1:]))]
    xp_j = [sigma_j["x+j"] for sigma_j in envelope[1:]]
    xm_j = [sigma_j["x-j"] for sigma_j in envelope[1:]]
    plt.plot(t, xp_j, color=colors[0])
    plt.plot(t, xm_j, color=colors[0])
    plt.fill_between(t, xm_j, xp_j, color=colors[1])
    plt.xlabel("Time (ms)")
    plt.ylabel("Cummulative Accesses")
    plt.title(title)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()


def convert_bandwidth_to_budget(
    bandwidth, weight_factor=1000, sample_rate_ns=10_000, cache_line_size=64
):
    return math.floor(
        bandwidth * weight_factor * sample_rate_ns / (1000 * cache_line_size)
    )


def convert_bandwidth_to_memguard_budget(
    bandwidth, weight_factor=1000, period_ns=1_000_000, cache_line_size=64
):
    return math.floor(bandwidth * weight_factor * period_ns / (1000 * cache_line_size))


def calculate_memguard_wcet(bm, envelope, budgets, period_ns=1_000_000):
    q = [
        convert_bandwidth_to_memguard_budget(
            i, weight_factor=1000, period_ns=period_ns, cache_line_size=64
        )
        for i in budgets
    ]
    wcet_regulated_ns = [
        get_wcet_cpu(bm.name, envelope, "CPU0", Q=i, P=period_ns, delta=10_000)
        for i in q
    ]
    wcet_regulated_memguard = [i / 1_000_000 for i in wcet_regulated_ns]
    return wcet_regulated_memguard


def calculate_mempol_wcet(bm, envelope, budgets, algo, algo_name):
    q = [
        convert_bandwidth_to_budget(
            i, weight_factor=1000, sample_rate_ns=10_000, cache_line_size=64
        )
        for i in budgets
    ]
    wcet_regulated_ns = [
        algo(bm.name, envelope, "CPU0", Q=i, P=1_000_000, delta=10_000, window_size=5)
        for i in q
    ]
    wcet_regulated_ms = [i / 1_000_000 for i in wcet_regulated_ns]
    return wcet_regulated_ms


def calculate_deviation(estimated, budget, measured):
    deviations = []
    deviation_percent = []
    for i in range(len(measured["measured_time_ms"])):
        index = budget.index(measured["measured_budget"][i])
        current_estimated = estimated[index]
        current_measured = measured["measured_time_ms"][i]
        deviations.append(current_estimated - current_measured)
        deviation_percent.append(
            (current_estimated - current_measured) / current_measured * 100
        )
    return deviations, deviation_percent


def calculate_deviation_and_plot_hist(
    estimated_list, budgets, measured, filename, title, legend
):
    deviations_list = []
    for estimated in estimated_list:
        deviations_list.append(calculate_deviation(estimated, budgets, measured))
    i = 0
    for deviations, deviation_percent in deviations_list:
        plt.hist(deviations, bins=50, label = legend[i], alpha=0.3,rasterized=True)
        plt.xlabel("Deviation (ms)")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc="upper right")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        i += 1
    plt.savefig(filename)
    plt.close()
    i = 0
    for deviations, deviation_percent in deviations_list:
        plt.hist(deviation_percent, bins=50, label = legend[i], alpha=0.3, rasterized=True)
        plt.xlabel("Deviation (%)")
        plt.ylabel("Frequency")
        plt.title(title)
        plt.legend(loc="upper right")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        i += 1
    plt.close()
    plt.savefig(filename.replace(".eps", "_perc.eps"))
    return

def calculate_deviation_and_plot(
    estimated_list, budgets, measured, filename, title, legend
):
    deviations_list = []
    for estimated in estimated_list:
        deviations_list.append(calculate_deviation(estimated, budgets, measured))
    i = 0
    for deviations, deviation_percent in deviations_list:
        plt.plot(
            measured["measured_budget"],
            deviations,
            linewidth=0.4,
            marker=".",
            label = legend[i]
        )
        plt.xlabel("Budget (MB/s)")
        plt.ylabel("Deviation (ms)")
        plt.title(title)
        plt.legend(loc="upper right")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        i += 1
    plt.savefig(filename)
    plt.close()
    return deviations, deviation_percent

def plot_measured_data_with_inf(
    wcet_list, budgets, data, filename, title="", legend=None
):
    if data != None:
        for i, wcet in enumerate(wcet_list):
            plt.plot(
                budgets,
                wcet,
                label=legend[i],
                linewidth=0.6,
            )
        for inf in data.keys():
            #random color
            random.randint(0, 255)
            color = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
            plt.plot(
                data[inf]["measured_budget"],
                data[inf]["measured_time_ms"],
                color=color,
                label=f"Inf: {inf}",
                linestyle="dashed",
                linewidth=0.4,
                
            )
        plt.legend(loc="upper right", prop={'size': 6})
        plt.xlabel("Budget (MB/s)")
        plt.ylabel("Execution Time (ms)")
        plt.title(title)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        plt.close()
    return


def main():
    args = parse_arguments()
    parser = BenchmarkParser(args)
    bm = parser.parse()
    envelope, wcet = get_envelope(
        bm.benchmark_dir, "CPU0", bm.jobs, weight_factor=[1000, 1000]
    )
    plot_run(
        bm.jobs,
        0,
        os.path.join(bm.benchmark_dir, "images", f"{bm.name}_run.eps"),
        title=f"{bm.name} Run",
    )
    plot_envelope(
        envelope,
        os.path.join(bm.benchmark_dir, "images", f"{bm.name}_envelope.eps"),
        title=f"{bm.name} Envelope",
    )
    budgets = range(100, 4001, 10)
    # create a list of memguard budgets and calculate wcet for each budget
    wcet_regulated_memguard = calculate_memguard_wcet(bm, envelope, budgets)
    # create a list of mempol budgets and calculate wcet for each budget
    wcet_regulated_mempol_sw = calculate_mempol_wcet(
        bm, envelope, budgets, get_wcet_cpu_mempol_sw, "sw"
    )
    wcet_regulated_mempol_tb = calculate_mempol_wcet(
        bm, envelope, budgets, get_wcet_cpu_mempol_tb, "tb"
    )
    plot_wcet_vs_measurement(
        [wcet_regulated_memguard, wcet_regulated_mempol_sw, wcet_regulated_mempol_tb],
        budgets,
        bm.measured_data,
        os.path.join(bm.benchmark_dir, "images", f"{bm.name}_estimated_wcet_vs_measurement.eps"),
        title=f"{bm.name} Estimated WCET vs Measurement",
        legend=["MG", "MP(SW)", "MP(TB)"],
    )
    plot_measured_data_with_inf(
        [wcet_regulated_memguard, wcet_regulated_mempol_sw, wcet_regulated_mempol_tb],
        budgets,
        bm.measured_data_with_inf,
        os.path.join(bm.benchmark_dir, "images", f"{bm.name}_measured_data_with_inf.eps"),
        title=f"{bm.name} Measured Data with Inf",
        legend=["MG", "MP(SW)", "MP(TB)"],
    )
    calculate_deviation_and_plot(
        [wcet_regulated_memguard, wcet_regulated_mempol_sw, wcet_regulated_mempol_tb],
        budgets,
        bm.measured_data,
        os.path.join(
            bm.benchmark_dir, "images", f"{bm.name}_deviation.eps"
        ),
        title=f"{bm.name} Mempol (TB) Deviation",
        legend=["MG", "MP(SW)", "MP(TB)"]
    )
    calculate_deviation_and_plot_hist(
        [wcet_regulated_memguard, wcet_regulated_mempol_sw, wcet_regulated_mempol_tb],
        budgets,
        bm.measured_data,
        os.path.join(
            bm.benchmark_dir, "images", f"{bm.name}_deviations_hist.eps"
        ),
        title=f"{bm.name} Memguard Deviation Histogram",
        legend=["MG", "MP(SW)", "MP(TB)"]
    )
    return


def plot_memory_budgets_comparison(
    memguard, mempol, budgets, measurement, filename, zoom=[0, 400], title=""
):
    ax1 = plt.subplot(212)
    ax2 = plt.subplot(221)
    ax1.axvspan(zoom[0], zoom[1], alpha=0.2, color="blue")
    ax1.plot(budgets, memguard)
    ax2.plot(budgets, memguard)
    ax1.plot(budgets, mempol, color="g")
    ax2.plot(budgets, mempol, color="g")
    ax1.scatter(
        measurement["measured_budget"], measurement["measured_time_ms"], color="r", s=3
    )
    ax2.scatter(
        measurement["measured_budget"], measurement["measured_time_ms"], color="r", s=3
    )
    plt.suptitle(title)

    ax1.set_ylabel("Execution Time (ms)")
    ax1.set_xlabel("Budget (MB/s)")
    ax2.set_xlim(zoom[0], zoom[1])
    # plt.show()
    plt.savefig(filename)
    plt.close()


def plot_wcet_vs_measurement(
    wcet_list, budgets, measurement, filename, title, legend
):
    for i, wcet in enumerate(wcet_list):
        plt.plot(budgets, wcet, label=legend[i], linewidth=0.4)
    
    plt.scatter(
        measurement["measured_budget"],
        measurement["measured_time_ms"],
        label="Measured",
        color="r",
        s=3,
        zorder=2,
    )
    plt.legend(loc="upper right")
    plt.xlabel("Budget (MB/s)")
    plt.ylabel("Execution Time (ms)")
    plt.title(title)
    plt.savefig(filename)
    # plt.show()
    plt.close()


def read_data():
    with open("parsed_data.json") as file:
        data = json.load(file)
    return data


def store_data(data):
    with open("parsed_data.json", "w") as file:
        json.dump(data, file, indent=4)


def parse_timings():
    result = []
    with open("timing.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            result.append(row)
    return result


def parse_timestamp():
    with open("canny.info", "r") as file:
        regex = re.compile(r"monotonic:(?P<timestamp>\d+.\d+)")
        result = []
        ts_prev = -1
        for line in file.readlines():
            ts = float(regex.search(line).group("timestamp"))
            if ts_prev == -1:
                ts_prev = float(regex.search(line).group("timestamp"))
            elif round(ts - ts_prev, 5) != 0.00256:
                print(line)
            result.append(ts)
            ts_prev = ts
    return result


def find_job(jobs, ts):
    for job in jobs:
        if job["start"] <= ts <= job["end"]:
            return job
    return None


def parse_data(limit=0):
    timings = parse_timings()
    data = {"jobs": [], "loop": [], "ts": [], "core": []}
    for job in timings:
        data["jobs"].append(
            {
                "start": float(job["period_start(seconds)"]),
                "end": float(job["job_end(seconds)"]),
                "ts": [],
                "pmu_read": [],
                "pmu_write": [],
            }
        )
    timestamps = parse_timestamp()
    current_job = None
    with open("canny.csv", "r") as file:
        limit_counter = 0
        initialized = False
        for line in file.readlines():
            limit_counter += 1
            if limit_counter >= limit and limit > 0:
                break
            num_cores = int((len(line.split(";")) - 1) / 3)
            if not initialized:
                initialized = True
                for coreid in range(num_cores):
                    data["core"].append(
                        {
                            "regulated": [],
                            "pmu_read": [],
                            "pmu_write": [],
                        }
                    )
            for i, element in enumerate(line.split(";")):
                if i == 0:
                    data["loop"].append(int(element))
                    ts = (
                        timestamps[int(math.floor(int(element) / 256))]
                        + (int(element) % 256) * 0.00001
                    )  # 10us
                    data["ts"].append(ts)
                    if current_job == None:
                        current_job = find_job(data["jobs"], ts)
                    if current_job != None:
                        if current_job["start"] <= ts <= current_job["end"]:
                            current_job["ts"].append(ts - current_job["start"])
                        else:
                            current_job = find_job(data["jobs"], ts)
                elif i <= num_cores:
                    data["core"][i - 1]["regulated"].append(int(element))
                else:
                    new_index = i - (num_cores + 1)
                    if new_index % 2 == 0:
                        coreid = int(new_index / 2)
                        data["core"][coreid]["pmu_read"].append(int(element))
                        if current_job != None and coreid == 0:
                            current_job["pmu_read"].append(int(element))
                    else:
                        coreid = int((new_index - 1) / 2)
                        data["core"][coreid]["pmu_write"].append(int(element))
                        if current_job != None and coreid == 0:
                            current_job["pmu_write"].append(int(element))
    return data


if __name__ == "__main__":
    main()
