import re
import sys
from collections import defaultdict, OrderedDict


class DecisionTreeConverter:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self.features = defaultdict(set)  # feature_name -> set of thresholds
        self.node_info = (
            {}
        )  # node_id -> {feature, threshold, samples, classification, etc}
        self.color_mapping = {
            "lean mixture": "lightyellow",
            "normal": "green",
            "rich mixture": "darkred",
            "low voltage": "darkorange",
        }

    def parse_dot_file(self, filename):
        """Parse DOT file and extract tree structure"""
        with open(filename, "r") as f:
            content = f.read()

        # Extract node definitions
        node_pattern = r'(\d+)\s+\[label="([^"]+)"\s*\]'
        for match in re.finditer(node_pattern, content):
            node_id = int(match.group(1))
            label = match.group(2)
            self.parse_node_label(node_id, label)

        # Extract edges
        edge_pattern = r"(\d+)\s*->\s*(\d+)"
        for match in re.finditer(edge_pattern, content):
            parent = int(match.group(1))
            child = int(match.group(2))
            self.edges.append((parent, child))

    def parse_node_label(self, node_id, label):
        """Extract information from node label"""
        lines = label.split("\\n")
        info = {"node_id": node_id}

        # Check if it's a decision node or leaf
        if "<=" in lines[0]:
            # Decision node
            decision_line = lines[0]
            feature_match = re.search(r"(.+?)\s*<=\s*([\d.]+)", decision_line)
            if feature_match:
                info["feature"] = feature_match.group(1).strip()
                info["threshold"] = float(feature_match.group(2))
                info["type"] = "decision"
                self.features[info["feature"]].add(info["threshold"])
        else:
            # Leaf node
            info["type"] = "leaf"
            # Extract classification from last line
            if len(lines) > 1:
                info["classification"] = lines[-1].strip()
            else:
                info["classification"] = lines[0].strip()

        self.node_info[node_id] = info

    def build_tree_structure(self):
        """Build parent-child relationships"""
        children = defaultdict(list)
        parents = {}

        for parent, child in self.edges:
            children[parent].append(child)
            parents[child] = parent

        return children, parents

    def get_decision_path(self, node_id, children, target_classification=None):
        """Get all decision paths from root to leaves"""
        paths = []

        def dfs(current, path, conditions):
            if self.node_info[current]["type"] == "leaf":
                classification = self.node_info[current]["classification"]
                paths.append(
                    {
                        "path": path[:],
                        "conditions": conditions[:],
                        "classification": classification,
                    }
                )
                return

            if current in children:
                feature = self.node_info[current]["feature"]
                threshold = self.node_info[current]["threshold"]

                # Left child (<=)
                if len(children[current]) > 0:
                    left_child = children[current][0]
                    path.append(f"{feature} <= {threshold}")
                    conditions.append((feature, "<=", threshold))
                    dfs(left_child, path, conditions)
                    path.pop()
                    conditions.pop()

                # Right child (>)
                if len(children[current]) > 1:
                    right_child = children[current][1]
                    path.append(f"{feature} > {threshold}")
                    conditions.append((feature, ">", threshold))
                    dfs(right_child, path, conditions)
                    path.pop()
                    conditions.pop()

        dfs(node_id, [], [])
        return paths

    def create_intervals(self, feature, thresholds):
        """Create interval notation from thresholds"""
        thresholds = sorted(list(thresholds))
        intervals = []

        if not thresholds:
            return [("(-∞, +∞)", None, None)]

        # First interval: (-∞, first_threshold]
        intervals.append((f"(-∞, {thresholds[0]}]", None, thresholds[0]))

        # Middle intervals: (prev_threshold, curr_threshold]
        for i in range(1, len(thresholds)):
            intervals.append(
                (
                    f"({thresholds[i-1]}, {thresholds[i]}]",
                    thresholds[i - 1],
                    thresholds[i],
                )
            )

        # Last interval: (last_threshold, +∞)
        intervals.append((f"({thresholds[-1]}, +∞)", thresholds[-1], None))

        return intervals

    def generate_nary_dot(self, output_filename):
        """Generate n-ary DOT file"""
        children, parents = self.build_tree_structure()

        # Find root (node with no parent)
        root = None
        for node_id in self.node_info:
            if node_id not in parents:
                root = node_id
                break

        if root is None:
            print("Error: Could not find root node")
            return

        # Get all decision paths
        paths = self.get_decision_path(root, children)

        # Group paths by first feature (root feature)
        root_feature = self.node_info[root]["feature"]
        root_thresholds = sorted(list(self.features[root_feature]))
        root_intervals = self.create_intervals(root_feature, root_thresholds)

        with open(output_filename, "w") as f:
            f.write("digraph NaryTree {\n")
            f.write('    node [shape=box, fontname="helvetica"];\n')
            f.write('    edge [fontname="helvetica"];\n')
            f.write("    rankdir=TB;\n\n")

            # Root node
            f.write(
                f'    root [label="{root_feature}", style=filled, fillcolor=lightgray, fontsize=14, fontweight=bold];\n\n'
            )

            node_counter = 1

            # Process each root interval
            for i, (interval_str, lower, upper) in enumerate(root_intervals):
                # Find paths that match this interval
                matching_paths = []

                for path in paths:
                    root_condition = (
                        path["conditions"][0] if path["conditions"] else None
                    )
                    if root_condition and root_condition[0] == root_feature:
                        feature, op, threshold = root_condition

                        # Check if this path matches the current interval
                        if self.condition_matches_interval(op, threshold, lower, upper):
                            matching_paths.append(path)

                if matching_paths:
                    # Create interval node
                    interval_node = f"interval_{i}"
                    f.write(f'    {interval_node} [label="{interval_str}"];\n')
                    f.write(f"    root -> {interval_node};\n")

                    # Process sub-features for this interval
                    self.process_sub_features(f, matching_paths, interval_node, 1)

            f.write("}\n")

    def condition_matches_interval(self, op, threshold, lower, upper):
        """Check if a condition matches an interval"""
        if op == "<=" and upper is not None and threshold == upper:
            return True
        elif op == ">" and lower is not None and threshold == lower:
            return True
        return False

    def process_sub_features(self, f, paths, parent_node, level):
        """Recursively process sub-features"""
        if not paths:
            return

        # Check if all paths lead to same classification
        classifications = [path["classification"] for path in paths]
        if len(set(classifications)) == 1:
            # All paths have same classification - create leaf
            classification = classifications[0]
            color = self.color_mapping.get(classification, "lightblue")
            fontcolor = "white" if color in ["green", "darkred"] else "black"

            leaf_node = f"leaf_{parent_node}_{level}"
            f.write(
                f'    {leaf_node} [label="{classification}", style=filled, fillcolor={color}'
            )
            if fontcolor == "white":
                f.write(f", fontcolor={fontcolor}")
            f.write("];\n")
            f.write(f"    {parent_node} -> {leaf_node};\n")
            return

        # Find next feature to split on
        next_features = defaultdict(list)
        for path in paths:
            if len(path["conditions"]) > level:
                feature = path["conditions"][level][0]
                next_features[feature].append(path)

        if not next_features:
            # Create leaves for remaining paths
            for i, path in enumerate(paths):
                classification = path["classification"]
                color = self.color_mapping.get(classification, "lightblue")
                fontcolor = "white" if color in ["green", "darkred"] else "black"

                leaf_node = f"leaf_{parent_node}_{level}_{i}"
                f.write(
                    f'    {leaf_node} [label="{classification}", style=filled, fillcolor={color}'
                )
                if fontcolor == "white":
                    f.write(f", fontcolor={fontcolor}")
                f.write("];\n")
                f.write(f"    {parent_node} -> {leaf_node};\n")
            return

        # Choose most common feature
        best_feature = max(next_features.keys(), key=lambda f: len(next_features[f]))

        # Create feature node
        feature_node = f"feature_{parent_node}_{level}"
        f.write(
            f'    {feature_node} [label="{best_feature}", style=filled, fillcolor=lightgray, fontweight=bold];\n'
        )
        f.write(f"    {parent_node} -> {feature_node};\n")

        # Process intervals for this feature
        feature_paths = next_features[best_feature]
        feature_thresholds = set()
        for path in feature_paths:
            if len(path["conditions"]) > level:
                threshold = path["conditions"][level][2]
                feature_thresholds.add(threshold)

        intervals = self.create_intervals(best_feature, feature_thresholds)

        for j, (interval_str, lower, upper) in enumerate(intervals):
            matching_paths = []

            for path in feature_paths:
                if len(path["conditions"]) > level:
                    condition = path["conditions"][level]
                    if self.condition_matches_interval(
                        condition[1], condition[2], lower, upper
                    ):
                        matching_paths.append(path)

            if matching_paths:
                interval_node = f"interval_{feature_node}_{j}"
                f.write(f'    {interval_node} [label="{interval_str}"];\n')
                f.write(f"    {feature_node} -> {interval_node};\n")

                self.process_sub_features(f, matching_paths, interval_node, level + 1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python tree_converter.py input.dot output.dot")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    converter = DecisionTreeConverter()
    converter.parse_dot_file(input_file)
    converter.generate_nary_dot(output_file)

    print(f"Converted {input_file} to {output_file}")


if __name__ == "__main__":
    main()
