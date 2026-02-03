# 🐝 PARL-Inspired Dynamic Agent Swarm for Novel Generation

A sophisticated self-directing agent swarm system inspired by **Parallel Agent Reinforcement Learning (PARL)** that dynamically creates specialized sub-agents with custom skills to generate complete novels.

## 🌟 Key Innovations

### 1. Dynamic Agent Creation
Unlike traditional fixed-pipeline systems, this swarm **creates agents on-the-fly** based on task requirements:
- The orchestrator analyzes each task
- Determines optimal agent configuration
- Creates custom skills if needed
- Spawns specialized agents dynamically

### 2. Parallel Execution
Inspired by PARL's parallel orchestration:
- Tasks are decomposed into parallelizable subtasks
- Independent tasks run concurrently
- Critical path optimization minimizes latency
- Up to 10 concurrent agents (configurable)

### 3. Self-Directing Swarm
No predefined workflows:
- Orchestrator decides task decomposition
- Agents can spawn sub-agents
- Skills are created dynamically based on needs
- System adapts to the specific novel requirements

### 4. Custom Skill System
Rich skill registry with dynamic creation:
```
SKILL CATEGORIES:
├── ANALYSIS      (deep_analysis, pattern_recognition, task_decomposition)
├── GENERATION    (prose_mastery, dialogue_craft, description_painting, action_choreography)
├── RESEARCH      (world_research, character_psychology, genre_expertise)
├── CREATIVITY    (plot_innovation, metaphor_weaving, emotional_engineering)
├── STRUCTURE     (narrative_architecture, scene_construction, chapter_design)
├── QUALITY       (continuity_tracking, style_consistency, editorial_polish)
├── SYNTHESIS     (integration_mastery, conflict_resolution)
└── SPECIALIZATION (dynamically created based on task)
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PARALLEL ORCHESTRATOR                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │    Task     │→ │   Agent     │→ │  Parallel   │                 │
│  │ Decomposer  │  │  Factory    │  │  Executor   │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
│         ↓                ↓                ↓                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    SKILL REGISTRY                            │   │
│  │  • Core Skills (20+ built-in)                                │   │
│  │  • Custom Skills (created dynamically)                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    DYNAMIC AGENT POOL                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │ Agent 4 │ │ Agent N │      │
│  │ (Brain  │ │ (Genre  │ │ (World  │ │ (Prose  │ │ (Custom │      │
│  │  Dump)  │ │ Expert) │ │ Builder)│ │ Writer) │ │  Role)  │      │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
│       ↑           ↑           ↑           ↑           ↑            │
│       └───────────┴───────────┴───────────┴───────────┘            │
│                    Parallel Execution Batches                       │
└─────────────────────────────────────────────────────────────────────┘
```

## 📊 Pipeline Stages

| Stage | Tasks | Parallelism | Description |
|-------|-------|-------------|-------------|
| 1. Foundation | Brain Dump, Genre Analysis, Themes | 3 parallel | Initial creative exploration |
| 2. Synopsis | Plot Structure | Sequential | Detailed story arc |
| 3. World Building | Characters, Locations, Systems | 3 parallel | Complete world bible |
| 4A. Outline | Chapter Structure | Sequential | 20-chapter outline |
| 4B. Scene Beats | Chapters 1-5, 6-10, 11-15, 16-20 | 4 parallel | Detailed scene breakdowns |
| 5. Writing | 4 batches of 5 chapters | 5 parallel per batch | Full prose generation |
| 6. QA | Continuity, Editorial | 2 parallel | Quality assurance |

## 🚀 Usage

### Requirements
```bash
pip install ollama
ollama pull qwen3:latest
```

### Run
```bash
# Interactive mode
python novel_swarm.py

# With topic
python novel_swarm.py "A hacker discovers a sentient city that rewrites reality using code"
```

### Configuration
Edit these at the top of `novel_swarm.py`:
```python
MODEL_NAME = "qwen3:latest"      # Ollama model
ENABLE_THINKING = True           # Show AI reasoning
OUTPUT_DIR = "novel_swarm_output"
MAX_PARALLEL_AGENTS = 10         # Max concurrent agents
MAX_TOTAL_AGENTS = 50            # Max total spawnable
ORCHESTRATOR_MAX_DEPTH = 5       # Max nesting for sub-orchestrators
```

## 🔍 Watching the Swarm

The system provides rich real-time feedback:

```
🐝 SWARM STATUS
│   Active Agents: 3  |  Total Spawned: 15  |  Completed: 12
│   Parallelism Factor: 2.50x

  🤖 SPAWN [a1b2c3d4] Character Psychology Expert (spawned by orchestrator)
     Skills: character_psychology, deep_analysis, emotional_engineering

  💭 [a1b2c3d4] Analyzing protagonist motivations based on backstory...

  ✓ DONE  [a1b2c3d4] Character Psychology Expert (45.3s)
```

## 📁 Output Files

```
novel_swarm_output/
├── complete_novel.md           # Full novel
├── chapter_01.md ... chapter_20.md  # Individual chapters
├── development_document.md     # All planning stages
└── swarm_documentation.json    # Complete agent logs
```

## 🧠 How Dynamic Agent Creation Works

1. **Task Analysis**: Orchestrator examines the task description
2. **Skill Matching**: LLM determines which skills are needed
3. **Custom Skill Creation**: If no existing skill fits, create one
4. **Agent Configuration**: Build system prompt with skill enhancements
5. **Execution**: Agent runs with full skill capabilities

Example of dynamically created agent:
```json
{
  "agent_type": "Cyberpunk Atmosphere Specialist",
  "skills": ["description_painting", "world_research", "genre_expertise"],
  "custom_skill": {
    "name": "neon_noir_aesthetics",
    "description": "Expertise in cyberpunk visual language",
    "prompt_enhancement": "You excel at describing rain-slicked streets, 
                          holographic advertisements, and the contrast 
                          between high-tech and urban decay..."
  }
}
```

## 📈 Metrics & Efficiency

The system tracks PARL-inspired metrics:

- **Total Agents Spawned**: How many agents were created
- **Critical Path Length**: Sequential steps (for latency calculation)
- **Parallel Efficiency**: `critical_path / total_steps` (higher = more parallel)
- **Task Completion Rate**: Success/failure tracking

Target: Achieve 2-3x parallel efficiency compared to sequential execution.

## 🔧 Extending the System

### Adding New Skills
```python
SKILL_REGISTRY.create_custom_skill(
    name="my_skill",
    description="What it does",
    category=SkillCategory.SPECIALIZATION,
    prompt_enhancement="Detailed instructions..."
)
```

### Custom Orchestration
```python
class MyOrchestrator(ParallelOrchestrator):
    def custom_workflow(self, task):
        subtasks = self.decompose_task(task, {})
        results = self.execute_parallel_group(subtasks, {})
        return results
```

## 🆚 Comparison: Sequential vs Swarm

| Metric | Sequential Pipeline | PARL Swarm |
|--------|---------------------|------------|
| Agent Types | Fixed 9 | Dynamic (20-50+) |
| Parallel Execution | None | Up to 10 concurrent |
| Custom Skills | No | Yes, created on-demand |
| Critical Path | ~20 steps | ~8 steps |
| Estimated Time | 2-3 hours | 45-90 minutes |

## 📝 License

MIT License - Use and modify freely!

## 🙏 Acknowledgments

- Inspired by PARL (Parallel Agent Reinforcement Learning)
- Built with [Ollama](https://ollama.com/) + [Qwen3](https://ollama.com/library/qwen3)
- Thinking mode for transparent AI reasoning
