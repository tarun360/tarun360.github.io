---
layout: post
title: 'AI discovers "X"'
description: 'AI discovers new science (with some help)'
tags: [ai, automated-science, alphaevolve]
# categories: [ai]
---

One thing I suspect is that claims about AI “discovering” new science or algorithms involve far more human agency than most researchers recognize.

Take AlphaEvolve [^1] as an example. It is advertised as an AI system capable of making novel discoveries — for instance, finding a faster method to multiply 4×4 complex-valued matrices. This is presented as an autonomous breakthrough by AI. But in reality, the process depended heavily on human agency.

AlphaEvolve was not simply told to invent a better algorithm for matrix multiplication. Instead, the problem was reformulated as a continuous optimization task, so that standard gradient-based methods like SGD or Adam could be applied. LLMs were then used to suggest hyperparameters (e.g., learning rate, weight decay), loss functions, and other heuristics. But this raises important questions that are often ignored: Why was this problem chosen? Why was it formulated in this particular way? The answer is that the authors drew on their understanding of LLMs and AlphaEvolve to identify problems where the system was most likely to succeed, and to frame those problems in a form it could handle. The paper even explicitly acknowledges that mathematicians (including Javier Gomez Serrano and Terence Tao) advised them on which problems to target and how best to formulate them:

“Most of these discoveries are on open problems suggested to us by external mathematicians Javier Gomez Serrano and Terence Tao, who also advised on how to best formulate them as inputs to AlphaEvolve.”

This invites comparison with earlier uses of computers in discovery. In 1976, Kenneth Appel and Wolfgang Haken used a computer to prove the four-colour conjecture — the result that any map can be coloured with only four colours such that no two adjacent regions share the same colour. Their program required hundreds of hours of computation, producing steps that no human could feasibly check in a lifetime. Yet the key to the proof was not an autonomous “discovery” by the computer. Rather, the human researchers played the key role by decomposing the problem into smaller pieces that a computer could handle, and then relying on brute-force computation.

So in both cases — AlphaEvolve and the four-colour proof — the human role was not incidental but central. The systems only produced results after researchers exercised agency in choosing which problems to tackle, how to represent them, and how to guide the computational process. This leads to the key question: in current claims that “AI discovered X,” what exactly is the role of human agency, and how should we understand the division of credit between humans and machines?

What could successful research to understand this more deeply look like? It is difficult to say. We would need a way to distinguish between dumb computation and creative computation — but that requires a definition of creativity, and at present I don’t see a clear path toward one. A more practical starting point might be to conduct case studies comparing modern AI-assisted discoveries (such as AlphaEvolve) with historical computational milestones (such as the four-colour conjecture). These studies could examine the extent to which creativity is attributed to computers rather than to human agency, and then present these insights to the research community to spark wider discussion about the division of roles in such discoveries.

[^1]: Novikov, Alexander, et al. “AlphaEvolve: A coding agent for scientific and algorithmic discovery.” arXiv preprint arXiv:2506.13131 (2025). [arXiv:2506.13131](https://arxiv.org/abs/2506.13131)


