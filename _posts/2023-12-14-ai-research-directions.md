---
layout: post
title: Research directions
date: 2023-12-14
description: Some worthwhile research directions
tags: research
categories: research 
disqus_comments: false
related_posts: true
---

<small>Last updated on: 4th Jan, 2024.</small>

This blog is inspired by the question posed by Dr. Richard Hamming in his famous “You and Your Research” talk: “If what you are working on is not important, and it’s not likely to lead to important things, then why are you working on it?”

I have two research directions in mind that I believe can lead to significant things in the field of AI (excluding applications in fields like health-care and biology, which I consider to be very promising): the first involves constructing a virtual playground for embodied multi-agent collaboration that can co-evolve into more intelligent agents, and the second focuses on building AI models that emulate the neural structures of the brain. I will briefly describe these two ideas below.

**Constructing a virtual playground for embodied multi-agent collaboration** 

Exciting advancements have been made in constructing virtual playgrounds to investigate single-agent exploration [1] and multi-agent collaboration [2, 3]. The latter, as per my exploration, seems to be less widely studied than the former. It is the latter that I believe holds more potential. These environments typically feature a leader agent and a follower agent working together to solve tasks. Leader-follower interactions can occur through various methods. For instance, the leader might issue instructions for a task, such as selecting a specific card from a set of distractor cards in the virtual environment, as in [2]. Subsequently, the follower is tasked with completing the assigned objective. Dialogue may transpire between the leader’s instructions and the follower’s actions. Notably, these instructions are typically communicated in a natural language, say English. These virtual environments are valuable for studying natural language interactions within collaborative and embodied environments.

One direction, which I believe could advance the research on embodied multi-agent collaboration, drawing inspiration from [2, 4, 5], and, to the best of my knowledge, less or un-explored thus far: Construct a virtual environment featuring a leader and a follower, where the follower’s role is to observe the leader executing a task and intelligently imitate them. For instance, if the leader hides behind a tree to evade a lion, the follower should seek shelter behind an available object, like a large stone nearest to it, to avoid the lion. Blindly copying the leader, such as traveling to the leader’s position and attempting to hide behind the same small tree, could harm both.

Initially, the follower may adopt an approach akin to behavior parsing used by apes [6], employing a mechanistic imitation model based on statistical correlation without a deep understanding of the leader’s motives. This is similar to the concept of feature expectation matching in inverse reinforcement learning [13] and apprenticeship learning [14]. However, as tasks become more difficult, there would be evolutionary pressure for the follower to develop a nuanced understanding of why a leader performs a specific task and determine the appropriate imitation variant, i.e., not doing behavior cloning [15] or feature expectation matching ― which is ape like-learning [6] ― but instead copying the meaning of the behavior, i.e., the meme [5] ― which is human-like learning. This necessitates creativity and the formulation of explanatory theories by the follower, and is one of the theories behind evolution from apes to humans [5].

I believe it would be advantageous for the follower to infer the leader’s mental state (Theory of Mind [7]) in order to generate explanatory theories about the reasons behind the leader’s actions and understand the meme. Furthermore, in this virtual environment, communication between the leader and follower could be facilitated through some communication channel, such as exchanging arrays of bits. Since communication between the leader and follower would be helpful for the follower to understand the meme, it’s conceivable to imagine this array of bits transforming into a rudimentary sign language (emergent communication [8]).

This is just an intuitive research direction that I believe may lead to something. Much remains to be addressed here. Firstly, the most challenging aspect is how to evolve the follower agent such that it doesn't rely on ape-like behavior parsing [6] (i.e., expected feature matching in inverse reinforcement learning parlance [13]), and instead learns to copy the meme. I suspect something along the lines of Theory of Mind and emergent communication paradigm may be useful for this. The second difficult aspect is creating a virtual environment that is <i>self-contained</i> with respect to explanations, i.e., the elements required by the agent to explain and understand the leader's meme are present in the environment itself. 

Given the difficulty of this task, as an iterative step to initiate progress, involving a human in designing meaningful tasks for the leader agent, that cannot be solved by behavior parsing, may be necessary. Perhaps, one can consider crowdsourcing to generate leader tasks, akin to [2], and then intelligently combine these tasks for more diversity. Nevertheless, this research direction seemed to me worthwhile to ponder upon.

**Building AI models that mimic the neural structures of the brain**

The present state of constructing increasingly complex machine learning architectures appears futile in creating a genuinely intelligent AI, or AGI. Merely combining different modalities blindly, such as outfitting a robot with vision, auditory, and text-understanding models, will not magically transform it into an intelligent agent. As articulated by Deutsch in [9], “Expecting to create an AGI without first understanding in detail how it works is like expecting skyscrapers to learn to fly if we build them tall enough.”

A more promising approach with a higher potential for AGI development would be to construct models inspired by the neural structures and geometries of the brain. Given that the brain inherently serves as an AGI, and with techniques such as fMRI, BOLD, and searchlight providing avenues for studying its internal structure, it is logical to examine the brain’s activity during diverse tasks. Subsequently, building AI models to replicate the observed neuronal structures during these tasks is a worthwhile strategy.

A noteworthy paper in this context is “Neural Knowledge Assembly in Humans and Neural Networks” [10]. This paper intrigued me due to the authors’ clever approach to testing a hypothesis (the elongation scheme) on rapid knowledge assembly in the human brain through a creative decision-making task performed by human participants. They monitored their brains’ dorsal stream structures using BOLD and searchlight techniques. Subsequently, the authors made a strikingly simple and elegant addition to a modest 2-layer neural network (a certainty matrix) to mimic the elongation scheme observed in human brains. In the current landscape of machine learning research, which often leans towards larger models, empirical fine-tuning, and explanation-less progress, it was invigorating to encounter a fundamental enhancement to a simple 2-layer neural network model inspired by knowledge assembly processes within the human brain.

Some other works include how speech aspects like context [11] and phonemes [12] impact neural structural representation. Understanding the representation structure of language in the human brain could aid in designing NLP models that emulate these structures, potentially resulting in improved NLP models similar to the approach taken in [10]. Overall, this fundamental research approach, where advancements in basic machine learning models are inspired by brain processes driven by conjecture and explanatory theories rather than solely increasing model complexity, is a worthwhile direction in pursuit of building better AI models.

**References**

[1] Zhu, Hao, et al. "EXCALIBUR: Encouraging and Evaluating Embodied Exploration." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[2] Sharf, Jacob, Mustafa Omer Gul, and Yoav Artzi. “CB2: Collaborative Natural Language Interaction Research Platform.” arXiv preprint arXiv:2303.08127 (2023).

[3] Suhr, Alane, et al. “Executing instructions in situated collaborative interactions.” arXiv preprint arXiv:1910.03655 (2019).

[4] Deutsch, D. (2011). The beginning of infinity: Explanations that transform the world. New York: Viking.

[5] Blackmore Susan J. The Meme Machine. Oxford University Press 1999.

[6] Byrne RW. Imitation as behaviour parsing. Philos Trans R Soc Lond B Biol Sci. 2003 Mar 29;358(1431):529-36. doi: 10.1098/rstb.2002.1219. PMID: 12689378; PMCID: PMC1693132.

[7] David Premack and Guy Woodruff. Does the chimpanzee have a theory of mind? Behavioral and brain sciences, 1(4):515–526, 1978.

[8] Lazaridou, Angeliki, and Marco Baroni. "Emergent multi-agent communication in the deep learning era." arXiv preprint arXiv:2006.02419 (2020).

[9] https://aeon.co/essays/how-close-are-we-to-creating-artificial-intelligence

[10] Nelli, Stephanie, et al. “Neural knowledge assembly in humans and neural networks.” Neuron 111.9 (2023): 1504-1516.

[11] Deniz, Fatma, et al. “Semantic representations during language comprehension are affected by context.” Journal of Neuroscience.

[12] Gong, Xue L., et al. “Phonemic segmentation of narrative speech in human cerebral cortex.” Nature Communications.

[13] A. Ng and S. Russell. “Algorithms for Inverse Reinforcement Learning”. In: Proceedings of the Seventeenth International Conference on Machine Learning. 2000, pp. 663–670

[14] P. Abbeel and A. Ng. “Apprenticeship Learning via Inverse Reinforcement Learning”. In: Proceedings of the TwentyFirst International Conference on Machine Learning. 2004

[15] Michie, D., Bain, M., & Hayes-Michie, J. E. (1990). Cognitive models from subcognitive skills. In M. Grimble, S. McGhee, & P. Mowforth (Eds.), Knowledge-based systems in industrial control. Stevenage: Peter Peregrinus.