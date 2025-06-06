import os
from collections import deque

def _remove_and_get_imports(file_content):
    """
    Removes import statements from the given file content.
    """
    imports = []
    # with open(file_path, 'r') as f:
    #     file_content = f.read()
    for line in file_content.splitlines():
        if line.startswith("import"):
            file_content = file_content.replace(line, "")
            imports.append(line[7:].strip())
    return (file_content.strip() + "\n", imports)

def load_repository_context(repo_path, file_name, theorem):
    """
    Returns the context of a theorem in a pair containing just the theorem's file (with imports) and everything in a single file that the theorem depends on.
    """
    if file_name.split("/")[0] == repo_path.split("/")[-1]:
        file_name = "/".join(file_name.split("/")[1:])

    file_path = os.path.join(repo_path, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r') as f:
        file_content = f.read()
    theorem = theorem.strip()
    if theorem.endswith("by"):
        theorem = theorem[:-2].strip()
    if theorem.endswith(":="):
        theorem = theorem[:-2].strip()
    # print(file_content)
    idx = file_content.find(theorem)
    if idx == -1:
        raise ValueError(f"Theorem '{theorem}' not found in file {file_name}.")

    current_file = file_content[:idx].strip() + "\n"

    # return current_file

    original_context, _ = _remove_and_get_imports(current_file)
    external_imports = []
    # for imp in original_imports:
    #     imp_path = os.path.join(repo_path, imp.replace('.', '/') + ".lean")
    #     if not os.path.exists(imp_path):
    #         external_imports.append("import " + imp)
    #     else:
    #         with open(imp_path, 'r') as f:
    #             current_file += "\n" + f.read()

    imports_graph = {file_path: set()}
    queue = [file_path]
    file_contents = {file_path: original_context}
    while queue:
        with open(queue[0], 'r') as f:
            # Read the file content and remove imports
            file_content = f.read()
        contents, imports = _remove_and_get_imports(file_content)
        if queue[0] not in file_contents:
            file_contents[queue[0]] = contents
        # imports_graph[queue[0]].update(imports)
        # print(imports_graph)
        for imp in imports:
            imp_path = os.path.join(repo_path, imp.replace('.', '/') + ".lean")
            if not os.path.exists(imp_path):
                external_imports.append("import " + imp)
            else:
                imports_graph[queue[0]].add(imp_path)
                if imp_path not in imports_graph:
                    imports_graph[imp_path] = set()
                    queue.append(imp_path)
        queue.pop(0)

    out = ""
    for file in topological_order(imports_graph):
        out = file_contents[file] + "\n\n" + out
    out = "\n".join(sorted(external_imports)) + "\n\n" + out

    # print(external_imports)

    return out


def topological_order(graph):
    """
    Given a directed acyclic graph as
        { node: {succ1, succ2, ...}, ... }
    returns a list of nodes in topological order,
    so if there's a path from u to v, u comes before v.
    """
    # compute in-degree for each node
    in_degree = {u: 0 for u in graph}
    for u, neighs in graph.items():
        for v in neighs:
            in_degree[v] = in_degree.get(v, 0) + 1

    # start with nodes of in-degree 0
    queue = deque([u for u, deg in in_degree.items() if deg == 0])
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, ()):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != len(in_degree):
        raise ValueError("Graph contains a cycle")
    return order


if __name__ == "__main__":
    # Example usage
    repo_path = "test-envs/minictx-v2/carleson"
    file_path = "carleson/Carleson/Discrete/ForestUnion.lean"
    theorem = """lemma C6_forest' (hkn : k ≤ n) :
    ℭ₆ (X := X) k n j = ⋃ l ∈ Iio (4 * n + 12), ⋃ u ∈ 𝔘₄ k n j l, 𝔗₂ k n j u :="""
    context = load_repository_context(repo_path, file_path, theorem)
    # print(len(context)//4)
    print(len(context)/297067)
