import math
from collections import Counter
import matplotlib.pyplot as plt

# --- Дані ---
data = [
    {"Outlook": "Sunny", "Temp": "Hot", "Humidity": "High", "Windy": False, "Play": "No"},
    {"Outlook": "Sunny", "Temp": "Hot", "Humidity": "High", "Windy": True, "Play": "No"},
    {"Outlook": "Overcast", "Temp": "Hot", "Humidity": "High", "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain", "Temp": "Mild", "Humidity": "High", "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain", "Temp": "Cool", "Humidity": "Normal", "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain", "Temp": "Cool", "Humidity": "Normal", "Windy": True, "Play": "No"},
    {"Outlook": "Overcast", "Temp": "Cool", "Humidity": "Normal", "Windy": True, "Play": "Yes"},
    {"Outlook": "Sunny", "Temp": "Mild", "Humidity": "High", "Windy": False, "Play": "No"},
    {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "Normal", "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain", "Temp": "Mild", "Humidity": "Normal", "Windy": False, "Play": "Yes"},
    {"Outlook": "Sunny", "Temp": "Mild", "Humidity": "Normal", "Windy": True, "Play": "Yes"},
    {"Outlook": "Overcast", "Temp": "Mild", "Humidity": "High", "Windy": True, "Play": "Yes"},
    {"Outlook": "Overcast", "Temp": "Hot", "Humidity": "Normal", "Windy": False, "Play": "Yes"},
    {"Outlook": "Rain", "Temp": "Mild", "Humidity": "High", "Windy": True, "Play": "No"},
]

target_attr = "Play"

# --- Алгоритм ID3 ---
def entropy(examples, target):
    cnt = Counter(example[target] for example in examples)
    total = len(examples)
    return sum(- (count/total) * math.log2(count/total) for count in cnt.values())

def split_by_attr(examples, attr):
    groups = {}
    for ex in examples:
        key = ex[attr]
        groups.setdefault(key, []).append(ex)
    return groups

def info_gain(examples, attr, target):
    base_entropy = entropy(examples, target)
    groups = split_by_attr(examples, attr)
    total = len(examples)
    remainder = sum(
        (len(group)/total) * entropy(group, target)
        for group in groups.values()
    )
    return base_entropy - remainder

def majority_class(examples, target):
    cnt = Counter(example[target] for example in examples)
    return cnt.most_common(1)[0][0]

def id3(examples, attrs, target):
    classes = [ex[target] for ex in examples]
    if len(set(classes)) == 1:
        return {"label": classes[0]}
    if not attrs:
        return {"label": majority_class(examples, target)}

    gains = {attr: info_gain(examples, attr, target) for attr in attrs}
    best_attr = max(gains, key=gains.get)

    tree = {"attribute": best_attr, "nodes": {}}
    groups = split_by_attr(examples, best_attr)
    remaining_attrs = [a for a in attrs if a != best_attr]

    for val, subset in groups.items():
        tree["nodes"][val] = id3(subset, remaining_attrs, target)
    return tree


# --- Візуалізація дерева ---
def draw_tree(node, x=0.5, y=1.0, dx=0.25, dy=0.12, ax=None, parent_pos=None, edge_label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_axis_off()
        draw_tree(node, ax=ax)
        plt.show()
        return

    if "label" in node:
        ax.text(x, y, node["label"], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", edgecolor="darkgreen"))
    else:
        attr = node["attribute"]
        ax.text(x, y, attr, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="navy"))
        n = len(node["nodes"])
        step = dx / max(n - 1, 1)
        for i, (val, child) in enumerate(node["nodes"].items()):
            child_x = x - dx/2 + i * step if n > 1 else x
            child_y = y - dy
            ax.plot([x, child_x], [y - 0.02, child_y + 0.05], color="black")
            ax.text((x + child_x)/2, (y + child_y)/2 + 0.02, str(val), fontsize=9, ha="center", color="darkred")
            draw_tree(child, child_x, child_y, dx/2, dy, ax=ax, parent_pos=(x, y), edge_label=val)

# --- Запуск ---
attribute_names = [a for a in data[0].keys() if a != target_attr]
tree = id3(data, attribute_names, target_attr)

print("Побудоване дерево рішень:\n")
def print_tree(node, indent=""):
    if "label" in node:
        print(indent + "-> " + node["label"])
    else:
        attr = node["attribute"]
        for val, child in node["nodes"].items():
            print(indent + f"[{attr} = {val}]")
            print_tree(child, indent + "   ")

print_tree(tree)

print("\nВізуалізація дерева...")
draw_tree(tree)


