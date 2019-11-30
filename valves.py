from dataclasses import field
from dataclasses import dataclass
from dataclasses import asdict
import random
import decimal
from decimal import Decimal
import argparse
import os
import math
import csv
import copy
import itertools
import functools
from munkres import Munkres
import tabulate


valve_count = 0
threshold: decimal.Decimal = None
quantiser: decimal.Decimal = None


def valve_counter():
    global valve_count
    valve_count += 1
    return valve_count


@dataclass(frozen=True)
class Valve:
    gap: Decimal
    shim_measured: Decimal
    spec_min: Decimal
    spec_max: Decimal
    number: int = field(default_factory=valve_counter)
    valve_type: str = None

    @property
    def perfect(self):
        return self.total - self.spec_max

    @property
    def total(self):
        return self.gap + self.shim_measured


@dataclass(frozen=True)
class IntakeValve(Valve):
    spec_min: Decimal = 0.10
    spec_max: Decimal = 0.15
    valve_type: str = "In"


@dataclass(frozen=True)
class ExhaustValve(Valve):
    spec_min: Decimal = 0.15
    spec_max: Decimal = 0.20
    valve_type: str = "Ex"


@functools.lru_cache()
def error(valve: Valve, proposed: Valve) -> float:
    return valve.perfect - proposed.shim_measured


@functools.lru_cache()
def error_percentage(valve: Valve, proposed: Valve) -> float:
    spec_range = valve.spec_max - valve.spec_min
    return 100 * error(valve, proposed) / spec_range


def score_one_pair(valve: Valve, donor: Valve) -> float:
    """
    we don't want something like least squares, we want to optimise
    any valves we reuse, and avoid penalising the case where they're massively 'out'
    """
    valve_error_percent = error_percentage(valve, donor)
    score = math.log(1 + min(abs(valve_error_percent), threshold))
    return float(score)


def score_pairs(valves: [(Valve, Valve)]) -> float:
    return sum(score_one_pair(valve, proposed) for valve, proposed in valves)


def search_permutations(valves: [Valve]) -> (float, [(Valve, Valve)]):
    perms = itertools.permutations(valves)
    best_score = 1e9
    best_pairs = None
    for perm_indx, perm in enumerate(perms):
        pairs = []
        for indx, valve in enumerate(valves):
            pairs.append((valve, perm[indx]))
        score = score_pairs(pairs)
        if score < best_score:
            best_score = score
            best_pairs = pairs
            print(best_pairs)
        if not perm_indx % 10000:
            print(perm_indx)
    return best_score, best_pairs


def search_by_order(valves: [Valve]) -> (float, [(Valve, Valve)]):
    # try to be more cleverer

    # shim distances
    all_distances = []
    for outer in valves:
        for inner in valves:
            all_distances.append([error(inner, outer), inner, outer])

    # reorder to list the smallest error pairing in order
    error_order = sorted(all_distances, key=lambda x: x[0], reverse=True)

    best_pair = []
    best_score = 1e9

    error_order_attempt = error_order[:]
    for i in range(10):
        remaining_valves = set(valves)
        set_valves = set()
        pairs = []
        for err, inner, outer in error_order_attempt:
            if inner in set_valves:
                continue
            if outer in remaining_valves:
                remaining_valves.remove(outer)
                set_valves.add(inner)
                pairs.append((inner, outer))
            if not remaining_valves:
                break

        score = score_pairs(pairs)
        if score < best_score:
            best_score = score
            best_pair = pairs

        random.shuffle(error_order_attempt[:i])
        print(error_order_attempt[:4])

    return best_score, best_pair


def search_by_hungarians(valves: [Valve]) -> (float, [(Valve, Valve)]):
    # try to be super clever

    # cost matrix
    all_distances = []
    for outer in valves:
        row = []
        for inner in valves:
            # error_value = error(inner, outer)
            # if error_value < 0:
            #     row.append(abs(error_value) * 10)
            # else:
            #     row.append(error_value ** 2)
            row.append(score_one_pair(inner, outer))
        all_distances.append(row)

    # docs at http://software.clapper.org/munkres/#usage
    m = Munkres()
    output = m.compute(all_distances)

    best_pair = []
    best_score = 0
    for row, col in output:
        best_pair.append((valves[row], valves[col]))
        value = all_distances[row][col]
        best_score += value
    return best_score, best_pair


def figure_out_quantised_purchase(valve: Valve) -> float:
    v = quantiser * round(decimal.Decimal(valve.perfect / float(quantiser)))
    return v.quantize(quantiser)

def run(valves):
    # best_score, best_pairs = search_permutations(valves)
    # best_score, best_pairs = search_by_order(valves)
    best_score, best_pairs = search_by_hungarians(valves)

    print(f"score: {best_score:.0f}")

    valves_to_buy_for = {}
    for a, b in sorted(best_pairs, key=lambda x: x[0].number):
        ep = error_percentage(a, b)
        if abs(ep) > threshold:
            valves_to_buy_for[a] = figure_out_quantised_purchase(a)

    print(f"\npurchase list ({len(valves_to_buy_for)} shims):")
    unique_sizes = {}
    for _, size in valves_to_buy_for.items():
        unique_sizes[size] = unique_sizes.setdefault(size, 0) + 1
    for size, count in sorted(unique_sizes.items()):
        print(f"{count: 2} x {size}")

    headers = [
        "valve",
        "side",
        "spec min",
        "spec max",
        "shim",
        "gap",
        "perfect shim",
        "error %",
        "swap with",
        "swap shim",
        "swap gap",
        "swap err %",
        "buy shim",
        "buy gap",
        "buy err %",
    ]
    content = []
    for a, b in sorted(best_pairs, key=lambda x: x[0].number):
        # should have had separate structures for buckets and shims ...
        buy_shim = float(figure_out_quantised_purchase(a))
        buy_gap = a.spec_max + (a.perfect - buy_shim)
        c = a.__class__(gap=buy_gap, shim_measured=buy_shim)
        content.append([
        a.number,
        a.valve_type,
        a.spec_min,
        a.spec_max,
        a.shim_measured,
        a.gap,
        a.perfect,
        int(round(error_percentage(a, a))),
        b.number,
        b.shim_measured,
        a.total - b.shim_measured,
        int(round(error_percentage(a, b))),
        c.shim_measured,
        c.gap,
        int(round(error_percentage(a, c))),
        ])
    print(tabulate.tabulate(content, headers, floatfmt=".3f"))
    
    with open("out.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([h.replace(" ", "_") for h in headers])
        for c in content:
            writer.writerow(c)

    # # FIXME
    # # now run through it all again with the new shims available
    # global valve_count
    # valve_count = max(v.number for v in valves)
    # for size in valves_to_buy_for.values():
    #     valves.append(IntakeValve(gap=0, shim_measured=float(size)))
    #
    # best_score, best_pairs = search_by_hungarians(valves)
    #
    # print(f"score: {best_score:.0f}")
    #
    # # figure out switches
    # print("\nin order of tolerance")
    # for a, b in sorted(best_pairs, key=lambda x: abs(error_percentage(*x))):
    #     print_valve_pair(a, b, valves_to_buy_for.get(a))


def demo_valves() -> [Valve]:
    return [
        ExhaustValve(gap=0.05, shim_measured=2.65),
        ExhaustValve(gap=0.06, shim_measured=2.65),
        ExhaustValve(gap=0.07, shim_measured=2.35),
        ExhaustValve(gap=0.08, shim_measured=2.65),
        ExhaustValve(gap=0.09, shim_measured=2.45),
        ExhaustValve(gap=0.10, shim_measured=2.45),
        ExhaustValve(gap=0.09, shim_measured=2.65),
        ExhaustValve(gap=0.09, shim_measured=2.65),
        IntakeValve(gap=0.09, shim_measured=2.75),
        IntakeValve(gap=0.09, shim_measured=2.75),
        IntakeValve(gap=0.09, shim_measured=2.85),
        IntakeValve(gap=0.09, shim_measured=2.85),
        IntakeValve(gap=0.09, shim_measured=2.55),
        IntakeValve(gap=0.09, shim_measured=2.55),
        IntakeValve(gap=0.09, shim_measured=2.65),
        IntakeValve(gap=0.09, shim_measured=2.65),
    ]


def load_valves(file_path) -> [Valve]:
    expanded = os.path.expanduser(file_path)
    full = os.path.abspath(expanded)
    fields = None
    content = []
    with open(full) as fh:
        reader = csv.DictReader(fh)
        for line in reader:
            content.append(line)
        fields = reader.fieldnames

    gap_field = None
    shim_field = None
    number_field = None
    intake_field = None
    for field in fields:
        if not gap_field and 'gap' in field.lower():
            gap_field = field
        if not shim_field and 'shim' in field.lower():
            shim_field = field
        if not number_field and 'valve' in field.lower():
            number_field = field
        if not intake_field and 'intake' in field.lower():
            intake_field = field

    valves = []
    for row in content:
        valve_type = IntakeValve if "intake" in row[intake_field].lower() else ExhaustValve
        valve = valve_type(
            gap=float(row[gap_field]),
            shim_measured=float(row[shim_field]),
            number=int(row[number_field]),
        )
        valves.append(valve)
    return valves


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store", type=str)
    parser.add_argument("--el", action="store", type=float, default=0.15)
    parser.add_argument("--eu", action="store", type=float, default=0.20)
    parser.add_argument("--il", action="store", type=float, default=0.10)
    parser.add_argument("--iu", action="store", type=float, default=0.15)
    parser.add_argument("--quantiser", action="store", type=decimal.Decimal, default=decimal.Decimal("0.05"))
    parser.add_argument("--threshold", action="store", type=decimal.Decimal, default=decimal.Decimal("50"))
    args = parser.parse_args()

    global quantiser
    quantiser = args.quantiser

    global threshold
    threshold = args.threshold

    default_input_path = os.path.join(os.path.dirname(__file__), "in.csv")
    if not args.csv and os.path.exists(default_input_path):
        args.csv = default_input_path

    if args.csv:
        valves = load_valves(args.csv)
    else:
        valves = demo_valves()
        print("WARNING USING FAKE DATA")

    # naaaasty hax to get custom limits into the already-instantiated objects
    valves_with_limits = []
    for valve in valves:
        current = asdict(valve)
        if isinstance(valve, ExhaustValve):
            current["spec_max"] = args.eu
            current["spec_min"] = args.el
            valves_with_limits.append(ExhaustValve(**current))
        if isinstance(valve, IntakeValve):
            current["spec_max"] = args.iu
            current["spec_min"] = args.il
            valves_with_limits.append(IntakeValve(**current))

    # TODO:
    # if you have spare shims lying around, enter them as fake valves with a large gap

    return run(valves_with_limits)


__name__ == "__main__" and cli()
