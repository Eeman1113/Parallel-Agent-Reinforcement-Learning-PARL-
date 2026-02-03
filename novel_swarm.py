#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                    PARL-INSPIRED DYNAMIC AGENT SWARM SYSTEM                              ║
║                                                                                          ║
║  A self-directing agent swarm that dynamically creates specialized sub-agents            ║
║  with custom skills, executing parallel workflows for novel generation.                  ║
║                                                                                          ║
║  Inspired by: Parallel Agent Reinforcement Learning (PARL)                               ║
║  - Dynamic agent instantiation without predefined roles                                  ║
║  - Parallel subtask execution with critical path optimization                            ║
║  - Trainable orchestrator with reward shaping                                            ║
║  - Up to N concurrent sub-agents with coordinated workflows                              ║
║                                                                                          ║
║  Key Innovation: Agents create OTHER agents with custom skills based on task analysis    ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import time
import uuid
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto
import textwrap
import copy
import hashlib

try:
    from ollama import chat, ChatResponse
except ImportError:
    print("Error: Ollama Python SDK not installed.")
    print("Install with: pip install ollama")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "qwen3:latest"
ENABLE_THINKING = True
OUTPUT_DIR = "novel_swarm_output"
MAX_PARALLEL_AGENTS = 10  # Maximum concurrent sub-agents
MAX_TOTAL_AGENTS = 50     # Maximum total agents that can be spawned
ORCHESTRATOR_MAX_DEPTH = 5  # Maximum nesting depth for sub-orchestrators
CRITICAL_PATH_WEIGHT = 0.7  # Weight for critical path vs total steps


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL STYLING
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    ORANGE = '\033[38;5;208m'
    PURPLE = '\033[38;5;129m'
    PINK = '\033[38;5;205m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Thread-safe print lock
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)

def print_banner(text: str, char: str = "═", width: int = 90):
    border = char * width
    safe_print(f"\n{Colors.CYAN}{border}{Colors.END}")
    safe_print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    safe_print(f"{Colors.CYAN}{border}{Colors.END}\n")

def print_swarm_status(active: int, total: int, completed: int, parallel_factor: float):
    safe_print(f"\n{Colors.ORANGE}╭{'─' * 70}╮{Colors.END}")
    safe_print(f"{Colors.ORANGE}│{Colors.END} {Colors.BOLD}🐝 SWARM STATUS{Colors.END}")
    safe_print(f"{Colors.ORANGE}│{Colors.END}   Active Agents: {Colors.GREEN}{active}{Colors.END}  |  Total Spawned: {Colors.BLUE}{total}{Colors.END}  |  Completed: {Colors.CYAN}{completed}{Colors.END}")
    safe_print(f"{Colors.ORANGE}│{Colors.END}   Parallelism Factor: {Colors.YELLOW}{parallel_factor:.2f}x{Colors.END}")
    safe_print(f"{Colors.ORANGE}╰{'─' * 70}╯{Colors.END}\n")

def print_agent_spawn(agent_id: str, agent_type: str, skills: list, parent: str = None):
    skill_str = ", ".join(skills[:3]) + ("..." if len(skills) > 3 else "")
    parent_str = f" (spawned by {parent})" if parent else ""
    safe_print(f"{Colors.GREEN}  🤖 SPAWN{Colors.END} [{agent_id[:8]}] {Colors.BOLD}{agent_type}{Colors.END}{parent_str}")
    safe_print(f"{Colors.DIM}     Skills: {skill_str}{Colors.END}")

def print_agent_complete(agent_id: str, agent_type: str, duration: float):
    safe_print(f"{Colors.CYAN}  ✓ DONE{Colors.END}  [{agent_id[:8]}] {agent_type} ({duration:.1f}s)")

def print_thinking_stream(agent_id: str, text: str):
    truncated = text[:100] + "..." if len(text) > 100 else text
    safe_print(f"{Colors.DIM}{Colors.MAGENTA}    💭 [{agent_id[:8]}] {truncated}{Colors.END}")


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL SYSTEM - Dynamic capabilities for agents
# ═══════════════════════════════════════════════════════════════════════════════

class SkillCategory(Enum):
    ANALYSIS = auto()
    GENERATION = auto()
    RESEARCH = auto()
    CREATIVITY = auto()
    STRUCTURE = auto()
    QUALITY = auto()
    SYNTHESIS = auto()
    SPECIALIZATION = auto()


@dataclass
class Skill:
    """Represents a capability that an agent can have"""
    name: str
    category: SkillCategory
    description: str
    prompt_enhancement: str  # Added to system prompt when skill is active
    prerequisites: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.name,
            "description": self.description
        }


class SkillRegistry:
    """Central registry of all available skills"""
    
    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self._register_core_skills()
    
    def _register_core_skills(self):
        """Register the built-in skill set"""
        
        core_skills = [
            # Analysis Skills
            Skill(
                name="deep_analysis",
                category=SkillCategory.ANALYSIS,
                description="Perform deep analytical reasoning on complex topics",
                prompt_enhancement="You excel at breaking down complex problems into components and analyzing each thoroughly. Use structured reasoning and consider multiple perspectives."
            ),
            Skill(
                name="pattern_recognition",
                category=SkillCategory.ANALYSIS,
                description="Identify patterns, themes, and connections across information",
                prompt_enhancement="You are skilled at finding hidden patterns, recurring themes, and unexpected connections. Look for both obvious and subtle relationships."
            ),
            Skill(
                name="task_decomposition",
                category=SkillCategory.ANALYSIS,
                description="Break complex tasks into parallelizable subtasks",
                prompt_enhancement="You excel at decomposing complex tasks into independent, parallelizable subtasks. Consider dependencies, critical paths, and optimal task distribution."
            ),
            
            # Generation Skills
            Skill(
                name="prose_mastery",
                category=SkillCategory.GENERATION,
                description="Write compelling, polished prose",
                prompt_enhancement="You are a master of prose. Write with vivid imagery, varied sentence structure, natural dialogue, and emotional resonance. Show, don't tell."
            ),
            Skill(
                name="dialogue_craft",
                category=SkillCategory.GENERATION,
                description="Create natural, distinctive character dialogue",
                prompt_enhancement="You specialize in dialogue that reveals character, advances plot, and sounds natural. Each character should have a distinctive voice."
            ),
            Skill(
                name="description_painting",
                category=SkillCategory.GENERATION,
                description="Create vivid, sensory descriptions",
                prompt_enhancement="You paint scenes with words using all five senses. Your descriptions are vivid but not purple, evocative but efficient."
            ),
            Skill(
                name="action_choreography",
                category=SkillCategory.GENERATION,
                description="Write dynamic action sequences",
                prompt_enhancement="You excel at writing action that is clear, dynamic, and exciting. Use short sentences for impact, maintain spatial awareness, and build tension."
            ),
            
            # Research Skills
            Skill(
                name="world_research",
                category=SkillCategory.RESEARCH,
                description="Research and develop fictional world details",
                prompt_enhancement="You are a world-building researcher. Consider geography, culture, history, economics, politics, and how they interconnect realistically."
            ),
            Skill(
                name="character_psychology",
                category=SkillCategory.RESEARCH,
                description="Deep psychological character analysis",
                prompt_enhancement="You understand human psychology deeply. Analyze motivations, fears, desires, defense mechanisms, and how trauma shapes behavior."
            ),
            Skill(
                name="genre_expertise",
                category=SkillCategory.RESEARCH,
                description="Expert knowledge of genre conventions and tropes",
                prompt_enhancement="You are an expert in genre conventions. Know the tropes, subvert expectations thoughtfully, and understand reader expectations."
            ),
            
            # Creativity Skills
            Skill(
                name="plot_innovation",
                category=SkillCategory.CREATIVITY,
                description="Generate innovative plot ideas and twists",
                prompt_enhancement="You generate unexpected yet logical plot developments. Subvert expectations while maintaining narrative coherence."
            ),
            Skill(
                name="metaphor_weaving",
                category=SkillCategory.CREATIVITY,
                description="Create meaningful metaphors and symbolism",
                prompt_enhancement="You weave metaphors and symbols throughout narrative. Create layers of meaning without being heavy-handed."
            ),
            Skill(
                name="emotional_engineering",
                category=SkillCategory.CREATIVITY,
                description="Craft emotional beats and resonance",
                prompt_enhancement="You engineer emotional experiences. Build to catharsis, create tension, deliver satisfying payoffs."
            ),
            
            # Structure Skills
            Skill(
                name="narrative_architecture",
                category=SkillCategory.STRUCTURE,
                description="Design story structure and pacing",
                prompt_enhancement="You architect narrative structure. Master three-act structure, rising action, climax, and resolution. Control pacing precisely."
            ),
            Skill(
                name="scene_construction",
                category=SkillCategory.STRUCTURE,
                description="Build individual scenes effectively",
                prompt_enhancement="You construct scenes with clear goals, conflict, and outcomes. Every scene must advance plot or character."
            ),
            Skill(
                name="chapter_design",
                category=SkillCategory.STRUCTURE,
                description="Design chapter flow and hooks",
                prompt_enhancement="You design chapters with strong openings, building middles, and compelling hooks. Control information reveal strategically."
            ),
            
            # Quality Skills
            Skill(
                name="continuity_tracking",
                category=SkillCategory.QUALITY,
                description="Maintain story continuity and consistency",
                prompt_enhancement="You track every detail for consistency. Names, timelines, character knowledge, object locations, and established facts."
            ),
            Skill(
                name="style_consistency",
                category=SkillCategory.QUALITY,
                description="Maintain consistent writing style and voice",
                prompt_enhancement="You maintain perfect style consistency. Match tone, vocabulary level, sentence rhythm, and narrative voice throughout."
            ),
            Skill(
                name="editorial_polish",
                category=SkillCategory.QUALITY,
                description="Final polish and refinement",
                prompt_enhancement="You are an expert editor. Remove redundancy, tighten prose, fix pacing issues, and enhance clarity."
            ),
            
            # Synthesis Skills
            Skill(
                name="integration_mastery",
                category=SkillCategory.SYNTHESIS,
                description="Integrate multiple elements cohesively",
                prompt_enhancement="You integrate disparate elements into cohesive wholes. Blend plot threads, character arcs, and themes seamlessly."
            ),
            Skill(
                name="conflict_resolution",
                category=SkillCategory.SYNTHESIS,
                description="Resolve conflicting information or approaches",
                prompt_enhancement="You resolve conflicts and contradictions. Find solutions that honor multiple constraints while maintaining quality."
            ),
        ]
        
        for skill in core_skills:
            self.register(skill)
    
    def register(self, skill: Skill):
        """Register a new skill"""
        self.skills[skill.name] = skill
    
    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name"""
        return self.skills.get(name)
    
    def get_by_category(self, category: SkillCategory) -> list[Skill]:
        """Get all skills in a category"""
        return [s for s in self.skills.values() if s.category == category]
    
    def create_custom_skill(self, name: str, description: str, category: SkillCategory, 
                           prompt_enhancement: str) -> Skill:
        """Dynamically create and register a new skill"""
        skill = Skill(
            name=name,
            category=category,
            description=description,
            prompt_enhancement=prompt_enhancement
        )
        self.register(skill)
        return skill
    
    def get_all_names(self) -> list[str]:
        return list(self.skills.keys())


# Global skill registry
SKILL_REGISTRY = SkillRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """Configuration for dynamically created agents"""
    agent_id: str
    agent_type: str
    role_description: str
    skills: list[str]
    system_prompt: str
    task: str
    parent_id: Optional[str] = None
    priority: int = 5  # 1-10, higher = more important
    estimated_steps: int = 1
    dependencies: list[str] = field(default_factory=list)  # Agent IDs this depends on


@dataclass
class AgentResult:
    """Result from an agent execution"""
    agent_id: str
    agent_type: str
    success: bool
    output: str
    thinking: str
    duration: float
    steps_taken: int
    spawned_agents: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SwarmMetrics:
    """Metrics for the agent swarm"""
    total_agents_spawned: int = 0
    active_agents: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_steps: int = 0
    critical_path_length: int = 0
    parallel_efficiency: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def update_efficiency(self):
        if self.total_steps > 0:
            self.parallel_efficiency = self.critical_path_length / self.total_steps


@dataclass
class TaskNode:
    """Represents a task in the execution graph"""
    task_id: str
    description: str
    agent_config: Optional[AgentConfig] = None
    result: Optional[AgentResult] = None
    children: list['TaskNode'] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class NovelState:
    """Shared state for the novel being generated"""
    topic: str
    genre: str = ""
    tone: str = ""
    style: str = ""
    pov: str = ""
    synopsis: str = ""
    world_building: dict = field(default_factory=dict)
    characters: list[dict] = field(default_factory=list)
    locations: list[dict] = field(default_factory=list)
    outline: list[dict] = field(default_factory=list)
    chapters: dict[int, str] = field(default_factory=dict)
    scene_beats: dict[int, list] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    # Locks for thread-safe updates
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def update(self, key: str, value: Any):
        with self._lock:
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.metadata[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if hasattr(self, key):
                return getattr(self, key)
            return self.metadata.get(key, default)


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicAgent:
    """
    A dynamically instantiated agent with custom skills.
    Can be created by the orchestrator or other agents.
    """
    
    def __init__(self, config: AgentConfig, skill_registry: SkillRegistry, model: str = MODEL_NAME):
        self.config = config
        self.skill_registry = skill_registry
        self.model = model
        self.last_thinking = ""
        self.last_response = ""
        self.steps_taken = 0
        self.spawned_agents: list[str] = []
    
    def _build_system_prompt(self) -> str:
        """Build the complete system prompt including skills"""
        base_prompt = self.config.system_prompt
        
        # Add skill enhancements
        skill_prompts = []
        for skill_name in self.config.skills:
            skill = self.skill_registry.get(skill_name)
            if skill:
                skill_prompts.append(f"[{skill.name.upper()}]: {skill.prompt_enhancement}")
        
        if skill_prompts:
            skills_section = "\n\nYOUR SPECIALIZED SKILLS:\n" + "\n".join(skill_prompts)
            base_prompt += skills_section
        
        return base_prompt
    
    def execute(self, context: dict = None) -> AgentResult:
        """Execute the agent's task"""
        start_time = time.time()
        
        system_prompt = self._build_system_prompt()
        
        # Build the user message with task and context
        user_message = f"TASK: {self.config.task}"
        if context:
            user_message += f"\n\nCONTEXT:\n{json.dumps(context, indent=2, default=str)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Stream the response
            response_stream = chat(
                model=self.model,
                messages=messages,
                think=ENABLE_THINKING,
                stream=True,
            )
            
            thinking = ""
            content = ""
            
            for chunk in response_stream:
                if chunk.message.thinking:
                    thinking += chunk.message.thinking
                    # Periodically show thinking
                    if len(thinking) % 500 < 50:
                        print_thinking_stream(self.config.agent_id, chunk.message.thinking)
                if chunk.message.content:
                    content += chunk.message.content
            
            self.last_thinking = thinking
            self.last_response = content
            self.steps_taken += 1
            
            duration = time.time() - start_time
            
            return AgentResult(
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                success=True,
                output=content,
                thinking=thinking,
                duration=duration,
                steps_taken=self.steps_taken,
                spawned_agents=self.spawned_agents
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return AgentResult(
                agent_id=self.config.agent_id,
                agent_type=self.config.agent_type,
                success=False,
                output=f"Error: {str(e)}",
                thinking="",
                duration=duration,
                steps_taken=self.steps_taken
            )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT FACTORY - Creates specialized agents dynamically
# ═══════════════════════════════════════════════════════════════════════════════

class AgentFactory:
    """
    Factory that creates specialized agents based on task requirements.
    Uses LLM to determine optimal agent configuration.
    """
    
    def __init__(self, skill_registry: SkillRegistry, model: str = MODEL_NAME):
        self.skill_registry = skill_registry
        self.model = model
        self.created_agents: dict[str, AgentConfig] = {}
    
    def analyze_and_create(self, task_description: str, available_context: dict,
                          parent_id: str = None) -> AgentConfig:
        """
        Analyze a task and create an optimal agent configuration.
        """
        agent_id = str(uuid.uuid4())
        
        # Use LLM to determine agent configuration
        analysis_prompt = f"""You are an AI architect that designs specialized agents.

AVAILABLE SKILLS:
{json.dumps([s.to_dict() for s in self.skill_registry.skills.values()], indent=2)}

TASK TO ACCOMPLISH:
{task_description}

CONTEXT:
{json.dumps(available_context, indent=2, default=str)[:2000]}

Design an agent to accomplish this task. Respond in JSON format:
{{
    "agent_type": "descriptive name for this agent type",
    "role_description": "detailed description of what this agent does",
    "skills": ["skill1", "skill2", "skill3"],  // 2-5 most relevant skills from available list
    "system_prompt": "The complete system prompt for this agent",
    "priority": 5,  // 1-10 importance
    "estimated_steps": 1,  // how many LLM calls needed
    "custom_skill": {{  // OPTIONAL: create a new skill if needed
        "name": "unique_skill_name",
        "description": "what it does",
        "category": "SPECIALIZATION",
        "prompt_enhancement": "instructions for the agent"
    }}
}}

Select skills that best match the task. Create a custom skill only if existing ones are insufficient."""

        messages = [
            {"role": "system", "content": "You design AI agents. Respond only with valid JSON."},
            {"role": "user", "content": analysis_prompt}
        ]
        
        try:
            response = chat(
                model=self.model,
                messages=messages,
                think=ENABLE_THINKING,
                stream=False,
            )
            
            # Parse the response
            content = response.message.content
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                config_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Create custom skill if specified
            if "custom_skill" in config_data and config_data["custom_skill"]:
                custom = config_data["custom_skill"]
                self.skill_registry.create_custom_skill(
                    name=custom["name"],
                    description=custom["description"],
                    category=SkillCategory[custom.get("category", "SPECIALIZATION")],
                    prompt_enhancement=custom["prompt_enhancement"]
                )
                # Add to skills list
                if custom["name"] not in config_data["skills"]:
                    config_data["skills"].append(custom["name"])
            
            config = AgentConfig(
                agent_id=agent_id,
                agent_type=config_data["agent_type"],
                role_description=config_data["role_description"],
                skills=config_data["skills"],
                system_prompt=config_data["system_prompt"],
                task=task_description,
                parent_id=parent_id,
                priority=config_data.get("priority", 5),
                estimated_steps=config_data.get("estimated_steps", 1)
            )
            
            self.created_agents[agent_id] = config
            return config
            
        except Exception as e:
            # Fallback to generic agent
            safe_print(f"{Colors.YELLOW}Warning: Agent design failed, using fallback: {e}{Colors.END}")
            return AgentConfig(
                agent_id=agent_id,
                agent_type="Generic Task Agent",
                role_description="A general-purpose agent",
                skills=["deep_analysis", "prose_mastery"],
                system_prompt=f"You are an AI assistant. Complete the following task thoroughly and professionally.",
                task=task_description,
                parent_id=parent_id
            )
    
    def create_predefined(self, agent_type: str, task: str, skills: list[str],
                         system_prompt: str, parent_id: str = None) -> AgentConfig:
        """Create an agent with predefined configuration"""
        agent_id = str(uuid.uuid4())
        config = AgentConfig(
            agent_id=agent_id,
            agent_type=agent_type,
            role_description=f"Predefined {agent_type}",
            skills=skills,
            system_prompt=system_prompt,
            task=task,
            parent_id=parent_id
        )
        self.created_agents[agent_id] = config
        return config


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL ORCHESTRATOR - Coordinates the agent swarm
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelOrchestrator:
    """
    PARL-inspired orchestrator that:
    1. Decomposes tasks into parallelizable subtasks
    2. Dynamically instantiates specialized sub-agents
    3. Coordinates parallel execution
    4. Tracks critical path for latency optimization
    """
    
    def __init__(self, skill_registry: SkillRegistry, model: str = MODEL_NAME):
        self.skill_registry = skill_registry
        self.model = model
        self.agent_factory = AgentFactory(skill_registry, model)
        self.metrics = SwarmMetrics()
        self.novel_state = None
        self.executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_AGENTS)
        self.task_graph: dict[str, TaskNode] = {}
        self.results_queue = queue.Queue()
        self.documentation_log = []
        
        # For critical path calculation
        self.stage_depths: dict[str, int] = {}
    
    def decompose_task(self, task: str, context: dict, depth: int = 0) -> list[dict]:
        """
        Use LLM to decompose a complex task into parallelizable subtasks.
        Returns a list of subtask specifications.
        """
        if depth >= ORCHESTRATOR_MAX_DEPTH:
            return [{"task": task, "parallel_group": 0, "dependencies": []}]
        
        decomposition_prompt = f"""You are a task decomposition expert optimizing for PARALLEL execution.

MAIN TASK:
{task}

CURRENT CONTEXT:
{json.dumps(context, indent=2, default=str)[:3000]}

Decompose this task into subtasks that can be executed IN PARALLEL where possible.

Rules:
1. Identify which subtasks are INDEPENDENT and can run simultaneously
2. Identify which subtasks have DEPENDENCIES and must wait
3. Group independent tasks into parallel batches
4. Minimize the critical path (longest sequential chain)

Respond with JSON:
{{
    "analysis": "brief analysis of parallelization opportunities",
    "subtasks": [
        {{
            "id": "subtask_1",
            "description": "detailed subtask description",
            "parallel_group": 0,  // Tasks in same group run in parallel
            "dependencies": [],   // IDs of subtasks that must complete first
            "agent_type_hint": "suggested agent specialization",
            "estimated_complexity": "low/medium/high"
        }},
        ...
    ],
    "critical_path_length": 3,  // Number of sequential steps
    "parallelism_factor": 2.5   // Speedup from parallelization
}}

Aim for maximum parallelism while maintaining task coherence."""

        messages = [
            {"role": "system", "content": "You decompose tasks for parallel execution. Respond only with valid JSON."},
            {"role": "user", "content": decomposition_prompt}
        ]
        
        try:
            response = chat(
                model=self.model,
                messages=messages,
                think=ENABLE_THINKING,
                stream=False,
            )
            
            content = response.message.content
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                decomposition = json.loads(json_match.group())
                
                # Log the decomposition thinking
                self.documentation_log.append({
                    "type": "decomposition",
                    "task": task,
                    "thinking": response.message.thinking,
                    "result": decomposition,
                    "timestamp": datetime.now().isoformat()
                })
                
                return decomposition.get("subtasks", [{"task": task, "parallel_group": 0, "dependencies": []}])
            
        except Exception as e:
            safe_print(f"{Colors.YELLOW}Decomposition failed: {e}, treating as single task{Colors.END}")
        
        return [{"id": "single", "description": task, "parallel_group": 0, "dependencies": []}]
    
    def execute_subtask(self, subtask: dict, context: dict, parent_id: str = None) -> AgentResult:
        """Execute a single subtask by creating and running an agent"""
        
        # Create specialized agent for this subtask
        agent_config = self.agent_factory.analyze_and_create(
            task_description=subtask["description"],
            available_context=context,
            parent_id=parent_id
        )
        
        # Update metrics
        self.metrics.total_agents_spawned += 1
        self.metrics.active_agents += 1
        
        # Print spawn info
        print_agent_spawn(
            agent_config.agent_id,
            agent_config.agent_type,
            agent_config.skills,
            parent_id
        )
        
        # Create and execute agent
        agent = DynamicAgent(agent_config, self.skill_registry, self.model)
        result = agent.execute(context)
        
        # Update metrics
        self.metrics.active_agents -= 1
        if result.success:
            self.metrics.completed_tasks += 1
        else:
            self.metrics.failed_tasks += 1
        self.metrics.total_steps += result.steps_taken
        
        # Print completion
        print_agent_complete(agent_config.agent_id, agent_config.agent_type, result.duration)
        
        # Log result
        self.documentation_log.append({
            "type": "agent_execution",
            "agent_id": agent_config.agent_id,
            "agent_type": agent_config.agent_type,
            "skills": agent_config.skills,
            "task": subtask["description"],
            "thinking": result.thinking,
            "output": result.output[:5000],
            "success": result.success,
            "duration": result.duration,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def execute_parallel_group(self, subtasks: list[dict], context: dict, 
                               parent_id: str = None) -> list[AgentResult]:
        """Execute a group of subtasks in parallel"""
        
        if not subtasks:
            return []
        
        futures = []
        for subtask in subtasks:
            # Submit each subtask to the thread pool
            future = self.executor.submit(
                self.execute_subtask,
                subtask,
                context.copy(),
                parent_id
            )
            futures.append((subtask, future))
        
        # Collect results
        results = []
        for subtask, future in futures:
            try:
                result = future.result(timeout=600)  # 10 minute timeout per task
                results.append(result)
            except Exception as e:
                safe_print(f"{Colors.RED}Task failed: {subtask.get('id', 'unknown')}: {e}{Colors.END}")
                results.append(AgentResult(
                    agent_id="failed",
                    agent_type="failed",
                    success=False,
                    output=str(e),
                    thinking="",
                    duration=0,
                    steps_taken=0
                ))
        
        return results
    
    def orchestrate(self, main_task: str, initial_context: dict) -> dict:
        """
        Main orchestration loop that coordinates the entire swarm.
        """
        safe_print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 ORCHESTRATOR: Analyzing main task...{Colors.END}\n")
        
        # Decompose the main task
        subtasks = self.decompose_task(main_task, initial_context)
        
        # Group subtasks by parallel group
        parallel_groups: dict[int, list[dict]] = {}
        for subtask in subtasks:
            group = subtask.get("parallel_group", 0)
            if group not in parallel_groups:
                parallel_groups[group] = []
            parallel_groups[group].append(subtask)
        
        # Execute groups in order (groups execute in parallel internally)
        all_results = []
        context = initial_context.copy()
        
        for group_num in sorted(parallel_groups.keys()):
            group_tasks = parallel_groups[group_num]
            
            safe_print(f"\n{Colors.YELLOW}═══ Executing Parallel Group {group_num} ({len(group_tasks)} tasks) ═══{Colors.END}\n")
            
            # Execute all tasks in this group in parallel
            results = self.execute_parallel_group(group_tasks, context)
            all_results.extend(results)
            
            # Update context with results
            for i, result in enumerate(results):
                if result.success:
                    context[f"result_{group_num}_{i}"] = result.output[:2000]
            
            # Update critical path
            self.metrics.critical_path_length += 1
            
            # Show swarm status
            print_swarm_status(
                self.metrics.active_agents,
                self.metrics.total_agents_spawned,
                self.metrics.completed_tasks,
                self.metrics.total_agents_spawned / max(1, self.metrics.critical_path_length)
            )
        
        # Calculate final metrics
        self.metrics.update_efficiency()
        
        return {
            "results": all_results,
            "metrics": self.metrics,
            "context": context
        }


# ═══════════════════════════════════════════════════════════════════════════════
# NOVEL GENERATION SWARM - Specialized orchestrator for novel writing
# ═══════════════════════════════════════════════════════════════════════════════

class NovelSwarmOrchestrator(ParallelOrchestrator):
    """
    Specialized orchestrator for novel generation that understands
    the novel creation pipeline and optimizes for it.
    """
    
    def __init__(self):
        super().__init__(SKILL_REGISTRY, MODEL_NAME)
        self.novel_state = NovelState(topic="")
    
    def generate_novel(self, topic: str) -> NovelState:
        """
        Generate a complete novel using the agent swarm.
        """
        self.novel_state = NovelState(topic=topic)
        
        print_banner("🐝 NOVEL GENERATION SWARM ACTIVATED 🐝", "═", 90)
        safe_print(f"{Colors.BOLD}Topic:{Colors.END} {topic}\n")
        
        start_time = time.time()
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Foundation (Brain Dump, Genre, Synopsis) - PARALLEL
        # ═══════════════════════════════════════════════════════════════
        print_banner("STAGE 1: Foundation Building", "─", 70)
        
        foundation_tasks = [
            {
                "id": "brain_dump",
                "description": f"""Create a comprehensive creative brainstorm for a novel about: "{topic}"
                
Include:
- Key themes and deeper meanings
- Core conflicts (internal and external)
- Potential plot twists (at least 5)
- Possible endings (happy, tragic, bittersweet)
- Mood and tone possibilities
- Unique narrative hooks
- Emotional resonance points
- Symbolic elements and metaphors

Be creative, bold, and thorough.""",
                "parallel_group": 0,
                "dependencies": []
            },
            {
                "id": "genre_analysis",
                "description": f"""Analyze the topic "{topic}" and determine:

1. Best PRIMARY GENRE (Science Fiction, Fantasy, Thriller, Literary Fiction, Horror, Romance, etc.)
2. SUBGENRE (Cyberpunk, Urban Fantasy, Psychological Thriller, etc.)
3. TONE (dark, hopeful, humorous, gritty, melancholic, whimsical)
4. WRITING STYLE (literary, cinematic, minimalist, poetic, pulpy)
5. NARRATIVE POV (1st person, 3rd limited, 3rd omniscient, multiple POV)

Explain why each choice serves this particular story.""",
                "parallel_group": 0,
                "dependencies": []
            },
            {
                "id": "initial_themes",
                "description": f"""Identify the core thematic elements for a novel about: "{topic}"

Determine:
- Primary theme (the main message/question)
- Secondary themes (2-3 supporting themes)
- Character themes (what main character learns)
- Societal themes (if applicable)
- How themes can be embodied through plot and character

Be specific about how these themes will manifest in the narrative.""",
                "parallel_group": 0,
                "dependencies": []
            }
        ]
        
        foundation_results = self.execute_parallel_group(
            foundation_tasks, 
            {"topic": topic}
        )
        
        # Store foundation results
        for i, result in enumerate(foundation_results):
            if result.success:
                if i == 0:
                    self.novel_state.update("brain_dump", result.output)
                elif i == 1:
                    self.novel_state.update("genre_analysis", result.output)
                    # Parse genre info
                    self._parse_genre(result.output)
                elif i == 2:
                    self.novel_state.update("themes", result.output)
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Synopsis Creation (uses foundation)
        # ═══════════════════════════════════════════════════════════════
        print_banner("STAGE 2: Synopsis Creation", "─", 70)
        
        synopsis_context = {
            "topic": topic,
            "brain_dump": self.novel_state.get("brain_dump", ""),
            "genre": self.novel_state.genre,
            "tone": self.novel_state.tone,
            "themes": self.novel_state.get("themes", "")
        }
        
        synopsis_task = {
            "id": "synopsis",
            "description": """Create a detailed novel synopsis including:

1. PROTAGONIST - Name, defining trait, core goal/desire, internal flaw
2. ANTAGONIST - Who/what opposes the protagonist
3. INCITING INCIDENT - The event that starts the story
4. FIRST ACT TURNING POINT - What locks the protagonist into the story
5. RISING ACTION - Key complications and escalations (at least 5)
6. MIDPOINT - The central pivot/revelation
7. SECOND ACT TURNING POINT - The crisis/dark moment
8. CLIMAX - The final confrontation
9. RESOLUTION - How the world/character changes
10. SUBPLOTS - 2-3 subplot threads with their arcs

Make it detailed enough to guide a 20-chapter novel.""",
            "parallel_group": 0,
            "dependencies": []
        }
        
        synopsis_results = self.execute_parallel_group([synopsis_task], synopsis_context)
        if synopsis_results and synopsis_results[0].success:
            self.novel_state.synopsis = synopsis_results[0].output
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: World Building (Character, Location, Systems) - PARALLEL
        # ═══════════════════════════════════════════════════════════════
        print_banner("STAGE 3: World Building", "─", 70)
        
        world_context = {
            "topic": topic,
            "synopsis": self.novel_state.synopsis,
            "genre": self.novel_state.genre,
            "tone": self.novel_state.tone
        }
        
        world_tasks = [
            {
                "id": "characters",
                "description": """Create detailed character profiles for ALL major characters (5-8 characters).

For EACH character provide:
- Full Name, Age, Role in story
- Physical description (distinctive features)
- Personality traits (5 specific traits)
- Strengths (3)
- Flaws/Weaknesses (3)
- Backstory (what shaped them)
- Secret/Hidden aspect
- Speech pattern/verbal tics
- Relationship to other characters
- Character arc (how they change)

Make characters feel real and distinct.""",
                "parallel_group": 0,
                "dependencies": []
            },
            {
                "id": "locations",
                "description": """Create detailed profiles for ALL key locations (5-7 locations).

For EACH location provide:
- Name and type
- Physical description (vivid sensory details)
- Atmosphere/mood
- Cultural/social context
- History/significance
- How it affects characters emotionally
- Key scenes that happen here
- Symbolic meaning (if any)

Make locations feel like characters themselves.""",
                "parallel_group": 0,
                "dependencies": []
            },
            {
                "id": "systems",
                "description": """Define the world's systems, rules, and mechanisms.

Include (as relevant to the story):
- Technology/Magic system (if applicable) - with clear rules and limits
- Social/Political structure
- Economic factors
- Cultural norms and taboos
- Historical context
- Any unique world mechanics

For each system define:
- How it works
- Its limitations/costs
- How it creates conflict
- How characters interact with it""",
                "parallel_group": 0,
                "dependencies": []
            }
        ]
        
        world_results = self.execute_parallel_group(world_tasks, world_context)
        
        for i, result in enumerate(world_results):
            if result.success:
                if i == 0:
                    self.novel_state.update("characters_raw", result.output)
                elif i == 1:
                    self.novel_state.update("locations_raw", result.output)
                elif i == 2:
                    self.novel_state.update("systems_raw", result.output)
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 4: Outline & Scene Beats - SEQUENTIAL then PARALLEL
        # ═══════════════════════════════════════════════════════════════
        print_banner("STAGE 4: Structural Outline", "─", 70)
        
        outline_context = {
            "topic": topic,
            "synopsis": self.novel_state.synopsis,
            "characters": self.novel_state.get("characters_raw", ""),
            "locations": self.novel_state.get("locations_raw", ""),
            "genre": self.novel_state.genre
        }
        
        outline_task = {
            "id": "chapter_outline",
            "description": """Create a detailed chapter-by-chapter outline for a 20-chapter novel.

For EACH chapter provide:
## Chapter [N]: [Evocative Title]
**Goal:** What this chapter must accomplish
**POV:** Whose perspective (if multiple POV)
**Key Events:**
- Event 1
- Event 2
- Event 3
**Emotional Arc:** The emotional journey
**Ends With:** The hook/tension leading to next chapter
**Foreshadowing:** What's planted or paid off

Structure:
- Chapters 1-5: Setup, character intro, inciting incident
- Chapters 6-10: Rising action, complications
- Chapters 11-14: Midpoint, escalation
- Chapters 15-18: Crisis, build to climax
- Chapters 19-20: Climax and resolution

Each chapter should be substantial and interconnected.""",
            "parallel_group": 0,
            "dependencies": []
        }
        
        outline_results = self.execute_parallel_group([outline_task], outline_context)
        if outline_results and outline_results[0].success:
            self.novel_state.update("outline_raw", outline_results[0].output)
        
        # Now create scene beats for multiple chapters in parallel
        print_banner("STAGE 4B: Scene Beats (Parallel)", "─", 70)
        
        # Split chapters into groups for parallel scene beat generation
        scene_beat_tasks = []
        for batch in range(4):  # 4 batches of 5 chapters each
            start_ch = batch * 5 + 1
            end_ch = min(start_ch + 4, 20)
            scene_beat_tasks.append({
                "id": f"scene_beats_{start_ch}_{end_ch}",
                "description": f"""Create detailed scene beats for Chapters {start_ch}-{end_ch}.

For each chapter, break it into 3-5 scenes. For each scene provide:

### Chapter [N], Scene [M]
**Summary:** One sentence
**Setting:** Location and time
**Characters:** Who is present
**Goal:** What the POV character wants
**Conflict:** What opposes them
**Outcome:** How it ends (success/failure/complication)
**Story Push:** How this advances the plot

Base this on the outline provided. Make scenes vivid and purposeful.""",
                "parallel_group": 0,
                "dependencies": []
            })
        
        scene_context = {
            "outline": self.novel_state.get("outline_raw", ""),
            "characters": self.novel_state.get("characters_raw", ""),
            "locations": self.novel_state.get("locations_raw", "")
        }
        
        scene_results = self.execute_parallel_group(scene_beat_tasks, scene_context)
        
        combined_scenes = ""
        for result in scene_results:
            if result.success:
                combined_scenes += result.output + "\n\n"
        self.novel_state.update("scene_beats_raw", combined_scenes)
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 5: Chapter Writing - PARALLEL BATCHES
        # ═══════════════════════════════════════════════════════════════
        print_banner("STAGE 5: Writing Chapters (Parallel Batches)", "─", 70)
        
        # Write chapters in parallel batches
        # Batch 1: Chapters 1-5 (setup)
        # Batch 2: Chapters 6-10 (rising action)  
        # Batch 3: Chapters 11-15 (midpoint/escalation)
        # Batch 4: Chapters 16-20 (climax/resolution)
        
        chapter_batches = [
            (1, 5, "Setup and Introduction"),
            (6, 10, "Rising Action"),
            (11, 15, "Midpoint and Escalation"),
            (16, 20, "Climax and Resolution")
        ]
        
        writing_base_context = {
            "topic": topic,
            "synopsis": self.novel_state.synopsis,
            "characters": self.novel_state.get("characters_raw", ""),
            "locations": self.novel_state.get("locations_raw", ""),
            "outline": self.novel_state.get("outline_raw", ""),
            "scene_beats": self.novel_state.get("scene_beats_raw", ""),
            "genre": self.novel_state.genre,
            "tone": self.novel_state.tone,
            "style": self.novel_state.style,
            "pov": self.novel_state.pov
        }
        
        previous_chapters_summary = ""
        
        for batch_start, batch_end, batch_name in chapter_batches:
            safe_print(f"\n{Colors.BOLD}{Colors.PURPLE}📖 Writing Batch: {batch_name} (Chapters {batch_start}-{batch_end}){Colors.END}\n")
            
            batch_tasks = []
            for ch_num in range(batch_start, batch_end + 1):
                batch_tasks.append({
                    "id": f"chapter_{ch_num}",
                    "description": f"""Write CHAPTER {ch_num} of the novel.

Requirements:
- Follow the outline and scene beats exactly
- Write 2500-3500 words
- Use the established style: {self.novel_state.style}
- Use the established POV: {self.novel_state.pov}
- Include rich sensory descriptions
- Write natural, distinctive dialogue
- End with a hook leading to the next chapter
- Reference earlier events naturally (if not chapter 1)

Format:
## Chapter {ch_num}: [Title from outline]

[Full chapter prose]

---

Write the complete chapter with professional quality prose.""",
                    "parallel_group": 0,
                    "dependencies": []
                })
            
            batch_context = writing_base_context.copy()
            batch_context["previous_chapters_summary"] = previous_chapters_summary
            batch_context["batch_info"] = f"Writing chapters {batch_start}-{batch_end}"
            
            batch_results = self.execute_parallel_group(batch_tasks, batch_context)
            
            # Store chapters and build summary for next batch
            for i, result in enumerate(batch_results):
                ch_num = batch_start + i
                if result.success:
                    self.novel_state.chapters[ch_num] = result.output
                    # Add to summary for continuity
                    previous_chapters_summary += f"\nChapter {ch_num}: {result.output[:500]}...\n"
            
            # Save progress after each batch
            self._save_progress()
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 6: Quality Assurance - PARALLEL
        # ═══════════════════════════════════════════════════════════════
        print_banner("STAGE 6: Quality Assurance", "─", 70)
        
        # Compile novel for review
        full_novel = "\n\n".join([
            self.novel_state.chapters.get(i, f"[Chapter {i} missing]")
            for i in range(1, 21)
        ])
        
        qa_tasks = [
            {
                "id": "continuity_check",
                "description": """Review the novel for CONTINUITY issues.

Check for:
- Character name consistency
- Timeline inconsistencies
- Location detail contradictions
- Object/item tracking errors
- Character knowledge consistency (do they know things they shouldn't?)
- Relationship continuity

List any issues found with chapter references and suggested fixes.""",
                "parallel_group": 0,
                "dependencies": []
            },
            {
                "id": "editorial_review",
                "description": """Provide an editorial assessment of the novel.

Evaluate:
- Overall plot coherence
- Character arc completeness
- Pacing effectiveness
- Dialogue quality
- Prose style consistency
- Opening hook strength
- Ending satisfaction
- Thematic resonance

Provide specific feedback with examples.""",
                "parallel_group": 0,
                "dependencies": []
            }
        ]
        
        qa_context = {
            "novel_sample": full_novel[:30000],  # First portion for review
            "characters": self.novel_state.get("characters_raw", ""),
            "synopsis": self.novel_state.synopsis
        }
        
        qa_results = self.execute_parallel_group(qa_tasks, qa_context)
        
        for i, result in enumerate(qa_results):
            if result.success:
                if i == 0:
                    self.novel_state.update("continuity_notes", result.output)
                elif i == 1:
                    self.novel_state.update("editorial_notes", result.output)
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL: Save everything
        # ═══════════════════════════════════════════════════════════════
        elapsed_time = time.time() - start_time
        
        self._save_final_output()
        self._save_documentation()
        
        # Print final statistics
        self._print_final_stats(elapsed_time)
        
        return self.novel_state
    
    def _parse_genre(self, genre_output: str):
        """Parse genre information from agent output"""
        lines = genre_output.lower().split('\n')
        for line in lines:
            if 'genre' in line and ':' in line:
                self.novel_state.genre = line.split(':', 1)[1].strip()[:50]
            elif 'tone' in line and ':' in line:
                self.novel_state.tone = line.split(':', 1)[1].strip()[:50]
            elif 'style' in line and ':' in line:
                self.novel_state.style = line.split(':', 1)[1].strip()[:50]
            elif 'pov' in line and ':' in line:
                self.novel_state.pov = line.split(':', 1)[1].strip()[:50]
    
    def _save_progress(self):
        """Save current progress"""
        ensure_output_directory()
        
        # Save individual chapters
        for ch_num, content in self.novel_state.chapters.items():
            filepath = os.path.join(OUTPUT_DIR, f"chapter_{ch_num:02d}.md")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _save_final_output(self):
        """Save the complete novel"""
        ensure_output_directory()
        
        # Build complete novel
        front_matter = f"""# {self.novel_state.topic}

**Genre:** {self.novel_state.genre}
**Tone:** {self.novel_state.tone}
**Style:** {self.novel_state.style}
**POV:** {self.novel_state.pov}

---

## Synopsis

{self.novel_state.synopsis}

---

"""
        
        chapters_text = "\n\n---\n\n".join([
            self.novel_state.chapters.get(i, f"[Chapter {i} not generated]")
            for i in range(1, 21)
        ])
        
        full_novel = front_matter + chapters_text
        
        filepath = os.path.join(OUTPUT_DIR, "complete_novel.md")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(full_novel)
        
        safe_print(f"\n{Colors.GREEN}✅ Complete novel saved to: {filepath}{Colors.END}")
    
    def _save_documentation(self):
        """Save all documentation and logs"""
        ensure_output_directory()
        
        # Save swarm documentation
        doc = {
            "topic": self.novel_state.topic,
            "metrics": {
                "total_agents_spawned": self.metrics.total_agents_spawned,
                "completed_tasks": self.metrics.completed_tasks,
                "failed_tasks": self.metrics.failed_tasks,
                "total_steps": self.metrics.total_steps,
                "critical_path_length": self.metrics.critical_path_length,
                "parallel_efficiency": self.metrics.parallel_efficiency
            },
            "novel_state": {
                "genre": self.novel_state.genre,
                "tone": self.novel_state.tone,
                "style": self.novel_state.style,
                "pov": self.novel_state.pov,
                "synopsis": self.novel_state.synopsis,
                "brain_dump": self.novel_state.get("brain_dump", ""),
                "characters": self.novel_state.get("characters_raw", ""),
                "locations": self.novel_state.get("locations_raw", ""),
                "systems": self.novel_state.get("systems_raw", ""),
                "outline": self.novel_state.get("outline_raw", ""),
                "scene_beats": self.novel_state.get("scene_beats_raw", ""),
                "continuity_notes": self.novel_state.get("continuity_notes", ""),
                "editorial_notes": self.novel_state.get("editorial_notes", "")
            },
            "agent_log": self.documentation_log
        }
        
        with open(os.path.join(OUTPUT_DIR, "swarm_documentation.json"), 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False, default=str)
        
        # Save human-readable development document
        dev_doc = f"""# Novel Development Document
## Generated by PARL-Inspired Agent Swarm

### Topic
{self.novel_state.topic}

### Swarm Metrics
- Total Agents Spawned: {self.metrics.total_agents_spawned}
- Completed Tasks: {self.metrics.completed_tasks}
- Failed Tasks: {self.metrics.failed_tasks}
- Total Steps: {self.metrics.total_steps}
- Critical Path Length: {self.metrics.critical_path_length}
- Parallel Efficiency: {self.metrics.parallel_efficiency:.2f}

### Genre & Style
- Genre: {self.novel_state.genre}
- Tone: {self.novel_state.tone}
- Style: {self.novel_state.style}
- POV: {self.novel_state.pov}

### Brain Dump
{self.novel_state.get("brain_dump", "Not generated")}

### Synopsis
{self.novel_state.synopsis}

### Characters
{self.novel_state.get("characters_raw", "Not generated")}

### Locations
{self.novel_state.get("locations_raw", "Not generated")}

### World Systems
{self.novel_state.get("systems_raw", "Not generated")}

### Chapter Outline
{self.novel_state.get("outline_raw", "Not generated")}

### Scene Beats
{self.novel_state.get("scene_beats_raw", "Not generated")}

### Continuity Notes
{self.novel_state.get("continuity_notes", "Not generated")}

### Editorial Assessment
{self.novel_state.get("editorial_notes", "Not generated")}

---

## Agent Activity Log

"""
        for entry in self.documentation_log:
            dev_doc += f"""
### {entry.get('type', 'unknown').upper()} - {entry.get('timestamp', '')}
**Agent:** {entry.get('agent_type', entry.get('agent_id', 'unknown'))}

**Task:** {entry.get('task', entry.get('description', ''))[:200]}...

**Thinking Preview:**
{entry.get('thinking', '')[:500]}...

**Output Preview:**
{entry.get('output', entry.get('result', ''))[:500] if isinstance(entry.get('output', entry.get('result', '')), str) else str(entry.get('output', entry.get('result', '')))[:500]}...

---
"""
        
        with open(os.path.join(OUTPUT_DIR, "development_document.md"), 'w', encoding='utf-8') as f:
            f.write(dev_doc)
        
        safe_print(f"{Colors.GREEN}✅ Documentation saved to: {OUTPUT_DIR}/{Colors.END}")
    
    def _print_final_stats(self, elapsed_time: float):
        """Print final statistics"""
        word_count = sum(len(ch.split()) for ch in self.novel_state.chapters.values())
        chapter_count = len(self.novel_state.chapters)
        
        print_banner("🎉 NOVEL GENERATION COMPLETE 🎉", "═", 90)
        
        safe_print(f"""
{Colors.BOLD}📊 SWARM STATISTICS:{Colors.END}
  ├─ Total Agents Spawned: {Colors.CYAN}{self.metrics.total_agents_spawned}{Colors.END}
  ├─ Tasks Completed: {Colors.GREEN}{self.metrics.completed_tasks}{Colors.END}
  ├─ Tasks Failed: {Colors.RED}{self.metrics.failed_tasks}{Colors.END}
  ├─ Total Steps: {self.metrics.total_steps}
  ├─ Critical Path: {self.metrics.critical_path_length}
  └─ Parallel Efficiency: {Colors.YELLOW}{self.metrics.parallel_efficiency:.2f}x{Colors.END}

{Colors.BOLD}📚 NOVEL STATISTICS:{Colors.END}
  ├─ Chapters Generated: {chapter_count}
  ├─ Total Word Count: {word_count:,}
  ├─ Estimated Pages: {word_count // 250}
  └─ Generation Time: {elapsed_time/60:.1f} minutes

{Colors.BOLD}📁 OUTPUT FILES:{Colors.END}
  ├─ {OUTPUT_DIR}/complete_novel.md
  ├─ {OUTPUT_DIR}/development_document.md
  ├─ {OUTPUT_DIR}/swarm_documentation.json
  └─ {OUTPUT_DIR}/chapter_XX.md (individual chapters)
""")


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    
    print(f"""
{Colors.CYAN}╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                            ║
║   ██████╗  █████╗ ██████╗ ██╗         ███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗       ║
║   ██╔══██╗██╔══██╗██╔══██╗██║         ██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║       ║
║   ██████╔╝███████║██████╔╝██║         ███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║       ║
║   ██╔═══╝ ██╔══██║██╔══██╗██║         ╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║       ║
║   ██║     ██║  ██║██║  ██║███████╗    ███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║       ║
║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝       ║
║                                                                                            ║
║              Parallel Agent Reinforcement Learning Inspired Novel Generator                ║
║                                                                                            ║
║   Features:                                                                                ║
║   • Dynamic agent creation with custom skills                                              ║
║   • Parallel subtask execution                                                             ║
║   • Critical path optimization                                                             ║
║   • Self-directing agent swarm                                                             ║
║   • Complete thinking visibility & documentation                                           ║
║                                                                                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝{Colors.END}
""")
    
    # Get topic from command line or prompt user
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        print(f"{Colors.YELLOW}Enter your novel topic (one line):{Colors.END}")
        print(f"{Colors.DIM}Example: 'A hacker discovers a sentient city that rewrites reality using code'{Colors.END}")
        print()
        topic = input(f"{Colors.BOLD}> {Colors.END}").strip()
    
    if not topic:
        print(f"{Colors.RED}Error: No topic provided. Exiting.{Colors.END}")
        sys.exit(1)
    
    # Create swarm orchestrator and generate novel
    swarm = NovelSwarmOrchestrator()
    
    try:
        novel_state = swarm.generate_novel(topic)
        print(f"\n{Colors.GREEN}🎉 Novel generation complete!{Colors.END}\n")
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Generation interrupted by user.{Colors.END}")
        print(f"{Colors.DIM}Partial output may be available in {OUTPUT_DIR}/{Colors.END}")
        swarm._save_progress()
        swarm._save_documentation()
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error during generation: {str(e)}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
