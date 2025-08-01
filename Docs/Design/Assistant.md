# Assistant

https://github.com/GuyTevet/CLoSD
https://human3daigc.github.io/Textoon_webpage/
https://huggingface.co/blog/wolfram/home-assistant
https://huggingface.co/tencent/Hunyuan3D-1
https://github.com/deepbeepmeep/Hunyuan3D-2GP
https://github.com/deepbeepmeep/mmgp
https://github.com/fagenorn/handcrafted-persona-engine
https://github.com/abdibrokhim/paper-ai-voice-assistant
https://arxiv.org/abs/2504.02197
https://github.com/yankooliveira/toads
https://github.com/kortix-ai/suna
https://github.com/joinly-ai/joinly
https://github.com/dyad-sh/dyad
https://github.com/openai/openai-cs-agents-demo
https://github.com/homanmirgolbabaee/arxiv-wizard-search?tab=readme-ov-file#-arxiv-research-assistant
https://www.youtube.com/watch?v=D7_ipDqhtwk
https://chatboxai.app/en
https://github.com/camel-ai/camel
https://github.com/h9-tec/agenttrace
https://github.com/Zerone-Laboratories/RIGEL
https://news.ycombinator.com/item?id=44301809
https://simonwillison.net/2024/Dec/20/building-effective-agents/
https://simonwillison.net/2025/Jun/2/claude-trace/
https://simonwillison.net/2025/Jun/14/multi-agent-research-system/
https://www.anthropic.com/engineering/built-multi-agent-research-system
https://arxiv.org/abs/2506.12915
    https://huggingface.co/datasets/PersonalAILab/PersonaFeedback
https://github.com/Roy3838/Observer
https://www.geoffreylitt.com/2025/04/12/how-i-made-a-useful-ai-assistant-with-one-sqlite-table-and-a-handful-of-cron-jobs
https://github.com/Pythagora-io/gpt-pilot
https://github.com/mierak/rmpc
https://github.com/e-p-armstrong/augmentoolkit
https://github.com/Fosowl/agenticSeek
https://huggingface.co/blog/python-tiny-agents
https://github.com/3-ark/Cognito-AI_Sidekick
https://github.com/Open-LLM-VTuber/Open-LLM-VTuber
https://github.com/moeru-ai/airi
https://github.com/HeavyNotFat/Agentic-AI-Desktop-Pet




https://github.com/BeehiveInnovations/zen-mcp-server
https://ghuntley.com/atlassian-rovo-source-code/
https://www.atlassian.com/blog/announcements/rovo-dev-command-line-interface
https://medium.com/@yugank.aman/top-agentic-ai-design-patterns-for-architecting-ai-systems-397798b44d5c
https://huggingface.co/eurecom-ds/Phi-3-mini-4k-socratic
https://giovannigatti.github.io/socratic-llm/
https://ceur-ws.org/Vol-3879/AIxEDU2024_paper_26.pdf
https://xata.io/blog/built-xata-mcp-server
https://qspeak.app/
https://github.com/Zie619/n8n-workflows
https://www.chatprd.ai/howiai
https://gerred.github.io/building-an-agentic-system/
https://forgecode.dev/blog/ai-agent-best-practices/
https://medium.com/data-science-collective/ten-lessons-from-a-year-building-ai-agents-in-legaltech-86a77f515757
https://jngiam.bearblog.dev/mcp-large-data/
https://ai.plainenglish.io/local-ai-automated-capture-building-your-own-knowledge-management-system-bbe583006927
https://docs.anthropic.com/en/docs/claude-code/common-workflows#run-parallel-claude-code-sessions-with-git-worktrees
https://arxiv.org/abs/2502.05589
https://tiamat.tsotech.com/pao
https://relay.md/
https://www.inkandswitch.com/essay/local-first/
```angular2html
**Core Architecture & Design Philosophy**

Anthropic’s system uses an **orchestrator-worker pattern** with these key components:

1. **LeadResearcher Agent**: Analyzes queries, develops strategies, spawns subagents
1. **Subagents**: Operate in parallel with separate context windows, performing focused searches
1. **CitationAgent**: Processes final results to ensure proper source attribution
1. **Dynamic iterative process**: Unlike static RAG systems, uses multi-step search that adapts based on findings

**Critical Success Factors**

1. **Token Economics**: Multi-agent systems use ~15x more tokens than regular chat (vs 4x for single agents)
1. **Performance Gains**: 90.2% improvement over single-agent Claude Opus 4
1. **Parallelization**: Up to 90% time reduction for complex queries
1. **Separation of Concerns**: Each subagent has distinct tools, prompts, and exploration trajectories

**Key Challenges Identified**

- Coordination complexity grows rapidly
- Agents can spawn excessive subagents (50 for simple queries)
- Risk of endless searching for nonexistent sources
- Agents distracting each other with excessive updates
- Synchronous execution creates bottlenecks
- Minor failures cascade into major behavioral changes

**Step-by-Step Plan for Creating Multi-Agent System Prompts**

**Phase 1: Foundation Architecture (Week 1-2)**

**Step 1: Define Agent Roles & Hierarchy**

1. Map out agent types:
- Orchestrator/Lead Agent
- Specialized Worker Agents (by domain/tool)
- Quality Control/Citation Agents
- Error Recovery Agents
1. Define clear boundaries:
- What each agent can/cannot do
- When to delegate vs handle internally
- Maximum agent spawn limits

**Step 2: Establish Communication Protocols**

1. Design message formats:
- Task assignment structure
- Result reporting format
- Error/status updates
1. Create handoff mechanisms:
- How agents pass work
- Context compression for handoffs
- Reference systems for artifacts

**Phase 2: Prompt Engineering Framework (Week 2-4)**

**Step 3: Lead Agent Prompt Template**

Core Components:

1. Role Definition
- “You are a research orchestrator responsible for…”
- Clear scope of authority
1. Decomposition Guidelines
- How to break complex queries
- Parallelizable vs sequential tasks
- Resource allocation rules
1. Delegation Framework
- Task description requirements:
  - Objective
  - Output format
  - Tool/source guidance
  - Clear boundaries
1. Scaling Rules (from simple to complex):
- Simple fact: 1 agent, 3-10 tool calls
- Comparison: 2-4 agents, 10-15 calls each
- Complex research: 10+ agents with divisions
1. Quality Control
- When to stop searching
- How to validate subagent work
- Synthesis requirements

**Step 4: Subagent Prompt Templates**

1. Task Reception
- “You receive specific research tasks…”
- How to interpret instructions
1. Search Strategy
- Start broad, then narrow
- Short queries (1-6 words initially)
- Progressive refinement
1. Tool Usage Heuristics
- Examine all available tools first
- Match tools to intent
- Prefer specialized over generic
1. Self-Evaluation
- Use interleaved thinking
- Assess result quality
- Identify gaps
- Know when complete
1. Result Formatting
- Structured output requirements
- Citation tracking
- Error reporting

**Phase 3: Tool Design & Integration (Week 3-4)**

**Step 5: Tool Interface Design**

1. Tool Descriptions
- Crystal clear purpose
- Distinct from other tools
- Usage examples
1. Error Handling
- Graceful failure modes
- Clear error messages
- Fallback options
1. Tool Selection Logic
- Decision trees for tool choice
- Context-based recommendations

**Step 6: Implement Thinking Mechanisms**

1. Extended Thinking Mode
- Planning scratch pad
- Strategy formulation
- Tool assessment
1. Interleaved Thinking
- Post-tool result analysis
- Gap identification
- Next step planning


**Phase 4: Coordination & State Management**

**Step 7: State Management System**

1. Checkpoint Design
- When to save state
- What to preserve
- Recovery mechanisms
1. Context Management
- Compression strategies
- External memory systems
- Fresh agent spawning
1. Error Recovery
- Resumption points
- Graceful degradation
- User communication

**Step 8: Parallel Execution Framework**

1. Synchronization Points
- When agents must wait
- When they can proceed
1. Result Aggregation
- Deduplication strategies
- Conflict resolution
- Quality ranking

**Phase 5: Evaluation & Iteration (Week 5-6)**

**Step 9: Create Evaluation Framework**

1. Small Sample Testing (20 queries)
- Representative use cases
- Edge case coverage
- Performance baselines
1. LLM-as-Judge Rubric
- Factual accuracy
- Citation accuracy
- Completeness
- Source quality
- Tool efficiency
1. Human Evaluation
- Behavioral edge cases
- Source selection biases
- User satisfaction

**Step 10: Self-Improvement Loop**

1. Agent Self-Analysis
- Give agents their prompts
- Let them diagnose failures
- Implement suggestions
1. Tool Description Refinement
- Test tools extensively
- Rewrite descriptions
- Measure improvement

**Phase 6: Production Hardening (Week 6-8)**

**Step 11: Monitoring & Observability**

1. Decision Pattern Tracking
- Agent choices
- Tool usage patterns
- Failure modes
1. Performance Metrics
- Token usage
- Time to completion
- Success rates

**Step 12: Deployment Strategy**

1. Rainbow Deployments
- Gradual rollout
- Version coexistence
- Safe rollback
1. A/B Testing Framework
- Prompt variations
- Architecture changes
- Performance comparison

**Key Prompt Engineering Principles**

**1. Think Like Your Agents**

- Build simulations to watch agents step-by-step
- Identify failure modes through observation
- Develop accurate mental models

**2. Teach Explicit Delegation**

- Never use vague instructions like “research X”
- Provide: objective, format, tools, boundaries
- Prevent duplicate work through clear division

**3. Scale Effort Appropriately**

- Embed scaling rules in prompts
- Match resources to query complexity
- Prevent overinvestment in simple tasks

**4. Start Wide, Then Narrow**

- Mirror expert human research patterns
- Begin with broad exploration
- Progressive focus refinement

**5. Guide the Thinking Process**

- Use extended thinking for planning
- Interleaved thinking for adaptation
- Make reasoning visible and controllable

**Advanced Considerations**

**Emergent Behaviors**

- Small prompt changes → large behavioral shifts
- Focus on collaboration frameworks over strict rules
- Define division of labor, not just tasks

**Asynchronous Future**

- Current systems are synchronous (bottleneck)
- Future: real-time coordination between agents
- Challenges: state consistency, error propagation

**Domain-Specific Adaptations**

- Research tasks: high parallelization potential
- Coding tasks: more sequential, less parallel
- Adjust architecture to domain characteristics

**Implementation Checklist**

□ Define clear agent roles and boundaries
□ Create detailed task decomposition rules
□ Design tool interfaces with clear descriptions
□ Implement thinking mechanisms (extended/interleaved)
□ Build state management and recovery systems
□ Create evaluation framework (automated + human)
□ Set up monitoring and observability
□ Design safe deployment strategies
□ Establish self-improvement loops
□ Test with small samples before scaling

This framework provides a comprehensive approach to building multi-agent systems based on Anthropic’s proven architecture. The key is to start simple, test rigorously, and iterate based on observed behaviors rather than assumptions about how agents “should” work.

```











@rogerarchwer - reddit
```
# Think

## Task

**THINK HARD and THINK HARDER and ULTRATHINK deeply about:** **$ARGUMENTS**

Use maximum thinking budget to analyze this problem systematically.

## Instructions

- Apply Carmack's systematic problem-solving methodology
- Think harder through multiple solution approaches
- Evaluate technical tradeoffs and practical implications
- Consider implementation complexity vs. real problem needs
- Provide methodical analysis before conclusions

## Approach

1. Break down the problem into core components
2. Identify the actual vs. perceived complexity
3. Consider both immediate and long-term consequences
4. Evaluate multiple solution paths
5. Recommend the most pragmatic approach
```