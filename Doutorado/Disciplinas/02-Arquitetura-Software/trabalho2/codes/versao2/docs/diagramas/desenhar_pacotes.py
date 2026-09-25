"""Gera o diagrama de pacotes UML da PIMUCS (requer o utilitário `dot` do Graphviz)."""
import subprocess

F = "Helvetica"
dot = f'''digraph Pacotes {{
  graph [fontname="{F}", rankdir=TB, nodesep=1.1, ranksep=1.1, pad=0.3];
  node  [shape=tab, style=filled, fontname="{F}", fontsize=14, margin="0.4,0.25"];
  edge  [style=dashed, arrowhead=vee, fontname="{F}", fontsize=11, color="#444444"];
  WebApplication [fillcolor="#e6d0e6", label="WebApplication\\n(pimucs.web_application)"];
  IAProcessing   [fillcolor="#dbe8fb", label="IAProcessing\\n(pimucs.ia_processing)"];
  Externo        [fillcolor="#ffe6cc", label="Sistema externo\\n(pimucs.external)"];
  DataManagement [fillcolor="#d5e8d4", label="DataManagement\\n(pimucs.data_management)"];
  Models         [fillcolor="#fffbd6", label="Models\\n(pimucs.models)"];
  {{ rank=same; WebApplication; IAProcessing; }}
  {{ rank=same; DataManagement; Externo; }}
  WebApplication -> DataManagement [label=" «use»"];
  IAProcessing   -> DataManagement [label=" «use»"];
  IAProcessing   -> Externo        [label=" «use»"];
  WebApplication -> Models         [label=" «use»"];
  IAProcessing   -> Models         [label=" «use»"];
  DataManagement -> Models         [label=" «use»"];
  Externo        -> Models         [label=" «use»"];
}}
'''
open("diagrama_pacotes.dot", "w", encoding="utf-8").write(dot)
for fmt in ("png", "svg", "pdf"):
    cmd = ["dot", f"-T{fmt}"] + (["-Gdpi=150"] if fmt == "png" else []) + ["diagrama_pacotes.dot", "-o", f"diagrama_pacotes.{fmt}"]
    subprocess.run(cmd, check=True)
print("ok")
