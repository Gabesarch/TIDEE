#<h1 align="left">
    Code for TIDEE:Novel Room Reorganization using Visuo-Semantic Common Sense Priors
</h1>

# TIDEE
TIDEE:Novel Room Reorganization using Visuo-Semantic Common Sense Priors
This repo contains code and data for TIDEE. If you like this paper, please cite us:
```
```

### Contents
<!--
# To create the table of contents, move the [TOC] line outside of this comment
# and then run the below Python block.
[TOC]
import markdown
with open("README.md", "r") as f:
    a = markdown.markdown(f.read(), extensions=["toc"])
    print(a[:a.index("</div>") + 6])
-->
<div class="toc">
<ul>
<li><a href="#-installation">Installation</a></li>
<li><a href="#-tidy-task"> Tidy Task </a></li>
<li><a href="#-rearrangement-task"> Rearrangement Task </a></li>
<!-- 
<li><a href="#%EF%B8%8F%EF%B8%8F-the-1--and-2-phase-tracks">☝️+✌️ The 1- and 2-Phase Tracks</a></li>
<li><a href="#-datasets">📊 Datasets</a></li>
</ul>
</li>
<li><a href="#-submitting-to-the-
    ">🛤️ Submitting to the Leaderboard</a></li>
<li><a href="#-allowed-observations">🖼️ Allowed Observations</a></li>
<li><a href="#-allowed-actions">🏃 Allowed Actions</a></li>
<li><a href="#-setting-up-rearrangement"> Setting up Rearrangement</a><ul> -->
<li><a href="#-tidy-task"> Tidy Task </a>
<ul>
<li><a href="#-detector"> Out-of-place Detector</a></li>
<li><a href="#-visual-memex"> Visual Memex</a></li>
<li><a href="#-visual-search-network"> Visual Search Network</a></li>
<li><a href="#-the-walkthrough-task-and-unshuffle-task-classes"> </a></li>
</ul>
</li>
<!-- <li><a href="#-object-poses">🗺️ Object Poses</a></li>
<li><a href="#-evaluation">🏆 Evaluation</a><ul>
<li><a href="#-when-are-poses-approximately-equal">📏 When are poses (approximately) equal?</a></li>
<li><a href="#-computing-metrics">💯 Computing metrics</a></li>
</ul>
</li>
<li><a href="#-training-baseline-models-with-allenact">🏋 Training Baseline Models with AllenAct</a><ul>
<li><a href="#-pretrained-models">💪 Pretrained Models</a></li>
</ul>
</li>
</ul>
</li>
<li><a href="#-citation">📄 Citation</a></li>
</ul> -->
</ul>
</div>

## Installation

