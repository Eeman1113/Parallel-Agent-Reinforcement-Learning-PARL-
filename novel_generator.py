#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MULTI-AGENT NOVEL GENERATION SYSTEM                       ║
║                                                                              ║
║  A sophisticated pipeline using multiple AI agents to generate complete      ║
║  120-200 page novels from a single topic line using Ollama + Qwen3-VL       ║
║                                                                              ║
║  Pipeline Stages:                                                            ║
║  1. Brain Dump Agent - Creative brainstorming                                ║
║  2. Genre & Style Agent - Determine genre, tone, POV                         ║
║  3. Synopsis Agent - Create detailed plot synopsis                           ║
║  4. World Builder Agent - Characters, locations, systems                     ║
║  5. Outline Architect Agent - Chapter-by-chapter outline                     ║
║  6. Scene Choreographer Agent - Detailed scene beats                         ║
║  7. Prose Writer Agent - Write full chapters                                 ║
║  8. Continuity Guardian Agent - Ensure consistency                           ║
║  9. Editor Agent - Polish and refine                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Optional, Generator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import textwrap

try:
    from ollama import chat, ChatResponse
except ImportError:
    print("Error: Ollama Python SDK not installed.")
    print("Install with: pip install ollama")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "qwen3:latest"  # Using qwen3 for thinking support
ENABLE_THINKING = True
OUTPUT_DIR = "novel_output"
MAX_RETRIES = 3
CHAPTERS_TARGET = 20  # Target number of chapters for 120-200 pages
WORDS_PER_CHAPTER = 3000  # Target words per chapter (~6-8 pages)


# ═══════════════════════════════════════════════════════════════════════════════
# STYLING & FORMATTING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_banner(text: str, char: str = "═", width: int = 80):
    """Print a formatted banner"""
    border = char * width
    print(f"\n{Colors.CYAN}{border}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(width)}{Colors.END}")
    print(f"{Colors.CYAN}{border}{Colors.END}\n")


def print_section(title: str):
    """Print a section header"""
    print(f"\n{Colors.YELLOW}{'─' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}▶ {title}{Colors.END}")
    print(f"{Colors.YELLOW}{'─' * 60}{Colors.END}\n")


def print_agent_start(agent_name: str, task: str):
    """Print agent activation notice"""
    print(f"\n{Colors.GREEN}╭{'─' * 58}╮{Colors.END}")
    print(f"{Colors.GREEN}│{Colors.END} {Colors.BOLD}🤖 AGENT: {agent_name}{Colors.END}")
    print(f"{Colors.GREEN}│{Colors.END} {Colors.DIM}Task: {task[:50]}...{Colors.END}")
    print(f"{Colors.GREEN}╰{'─' * 58}╯{Colors.END}\n")


def print_thinking(text: str):
    """Print thinking output with special formatting"""
    wrapped = textwrap.fill(text, width=76, initial_indent="  ", subsequent_indent="  ")
    print(f"{Colors.DIM}{Colors.MAGENTA}{wrapped}{Colors.END}", end="", flush=True)


def print_content(text: str):
    """Print content output"""
    print(f"{Colors.END}{text}", end="", flush=True)


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print a progress bar"""
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent = 100 * current / total
    print(f"\r{Colors.BLUE}{prefix}: [{bar}] {percent:.1f}% ({current}/{total}){Colors.END}", end="", flush=True)
    if current == total:
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Character:
    """Represents a character in the novel"""
    name: str
    age: str
    role: str
    personality_traits: list[str]
    strengths: list[str]
    flaws: list[str]
    backstory: str
    relationships: dict[str, str]
    arc: str


@dataclass
class Location:
    """Represents a location in the novel"""
    name: str
    description: str
    cultural_details: str
    environmental_details: str
    plot_relevance: str


@dataclass
class WorldSystem:
    """Represents a system/rule in the novel world"""
    name: str
    description: str
    rules: list[str]
    limitations: list[str]
    conflict_potential: str


@dataclass
class SceneBeat:
    """Represents a scene beat within a chapter"""
    scene_number: int
    summary: str
    setting: str
    characters: list[str]
    conflict: str
    outcome: str
    story_push: str


@dataclass
class ChapterOutline:
    """Represents a chapter outline"""
    number: int
    title: str
    goal: str
    key_events: list[str]
    emotional_beats: list[str]
    connections: str
    foreshadowing: list[str]
    scene_beats: list[SceneBeat] = field(default_factory=list)


@dataclass
class NovelBlueprint:
    """Complete blueprint for the novel"""
    topic: str
    brain_dump: str = ""
    genre: str = ""
    subgenre: str = ""
    tone: str = ""
    writing_style: str = ""
    pov: str = ""
    style_explanation: str = ""
    synopsis: str = ""
    characters: list[Character] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)
    world_systems: list[WorldSystem] = field(default_factory=list)
    chapter_outlines: list[ChapterOutline] = field(default_factory=list)
    chapters: dict[int, str] = field(default_factory=dict)
    

@dataclass
class DocumentationLog:
    """Tracks all agent activities and outputs"""
    entries: list[dict] = field(default_factory=list)
    
    def add_entry(self, agent: str, stage: str, thinking: str, output: str, timestamp: str = None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        self.entries.append({
            "timestamp": timestamp,
            "agent": agent,
            "stage": stage,
            "thinking": thinking,
            "output": output
        })
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# BASE AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, description: str, model: str = MODEL_NAME):
        self.name = name
        self.description = description
        self.model = model
        self.last_thinking = ""
        self.last_response = ""
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    @abstractmethod
    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        """Execute the agent's task and return updated blueprint"""
        pass
    
    def call_llm(self, messages: list[dict], stream: bool = True) -> tuple[str, str]:
        """
        Call the LLM and return (thinking, content) tuple.
        Streams output to terminal for visibility.
        """
        thinking_accumulated = ""
        content_accumulated = ""
        
        if stream:
            response_stream = chat(
                model=self.model,
                messages=messages,
                think=ENABLE_THINKING,
                stream=True,
            )
            
            in_thinking = False
            thinking_header_printed = False
            content_header_printed = False
            
            for chunk in response_stream:
                # Handle thinking
                if chunk.message.thinking:
                    if not thinking_header_printed:
                        print(f"\n{Colors.MAGENTA}💭 Thinking:{Colors.END}")
                        thinking_header_printed = True
                        in_thinking = True
                    thinking_accumulated += chunk.message.thinking
                    print_thinking(chunk.message.thinking)
                
                # Handle content
                if chunk.message.content:
                    if in_thinking or not content_header_printed:
                        print(f"\n\n{Colors.GREEN}📝 Response:{Colors.END}\n")
                        content_header_printed = True
                        in_thinking = False
                    content_accumulated += chunk.message.content
                    print_content(chunk.message.content)
            
            print()  # Final newline
            
        else:
            response = chat(
                model=self.model,
                messages=messages,
                think=ENABLE_THINKING,
                stream=False,
            )
            thinking_accumulated = response.message.thinking or ""
            content_accumulated = response.message.content or ""
            
            if thinking_accumulated:
                print(f"\n{Colors.MAGENTA}💭 Thinking:{Colors.END}")
                print(f"{Colors.DIM}{thinking_accumulated}{Colors.END}")
            if content_accumulated:
                print(f"\n{Colors.GREEN}📝 Response:{Colors.END}")
                print(content_accumulated)
        
        self.last_thinking = thinking_accumulated
        self.last_response = content_accumulated
        return thinking_accumulated, content_accumulated


# ═══════════════════════════════════════════════════════════════════════════════
# SPECIALIZED AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

class BrainDumpAgent(BaseAgent):
    """Agent responsible for initial creative brainstorming"""
    
    def __init__(self):
        super().__init__(
            name="Brain Dump Agent",
            description="Expands a single topic into a rich creative brainstorm"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a master creative writer and story conceptualist. Your role is to take a single-line topic and expand it into a comprehensive creative brainstorm.

You must explore:
1. KEY THEMES - What deeper meanings could this story explore?
2. CORE CONFLICT - What is the central struggle?
3. POTENTIAL TWISTS - What unexpected turns could occur?
4. POSSIBLE ENDINGS - Both happy, tragic, and bittersweet options
5. MOOD AND TONE IDEAS - How should the reader feel?
6. UNIQUE NARRATIVE HOOKS - What makes this story special?
7. EMOTIONAL RESONANCE - What universal human experiences does this touch?
8. SYMBOLIC ELEMENTS - What metaphors or symbols could enrich the narrative?

Be creative, bold, and thorough. Generate multiple options for each category.
Output in clear sections with headers."""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Expanding topic into creative brainstorm")
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Take this single topic and create a comprehensive creative brainstorm for a full novel:

TOPIC: {blueprint.topic}

Generate a detailed brain dump covering all aspects mentioned in your instructions. Be creative and explore multiple possibilities for each element."""}
        ]
        
        thinking, content = self.call_llm(messages)
        blueprint.brain_dump = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="Brain Dump",
            thinking=thinking,
            output=content
        )
        
        return blueprint


class GenreStyleAgent(BaseAgent):
    """Agent responsible for determining genre, tone, and writing style"""
    
    def __init__(self):
        super().__init__(
            name="Genre & Style Agent",
            description="Determines the optimal genre, tone, and writing approach"
        )
    
    def get_system_prompt(self) -> str:
        return """You are an expert literary analyst specializing in genre classification and writing style. Your role is to analyze a creative brainstorm and determine the optimal genre, subgenre, tone, writing style, and narrative POV for the novel.

You must determine:
1. GENRE - Primary genre (e.g., Science Fiction, Fantasy, Thriller, Literary Fiction, Horror, Romance)
2. SUBGENRE - More specific classification (e.g., Cyberpunk, Urban Fantasy, Psychological Thriller)
3. TONE - The emotional atmosphere (e.g., dark, hopeful, humorous, gritty, melancholic, whimsical)
4. WRITING STYLE - The prose approach (e.g., literary, cinematic, minimalist, poetic, pulpy, YA-accessible)
5. NARRATIVE POV - Point of view (1st person, 3rd person limited, 3rd person omniscient, multiple POV)

For each choice, provide a brief explanation of WHY this choice best serves the story.

Output format:
GENRE: [choice]
SUBGENRE: [choice]
TONE: [choice]
WRITING STYLE: [choice]
NARRATIVE POV: [choice]

EXPLANATION:
[Your reasoning for each choice]"""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Analyzing brainstorm for optimal genre and style")
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Analyze this creative brainstorm and determine the optimal genre, style, and POV:

ORIGINAL TOPIC: {blueprint.topic}

BRAIN DUMP:
{blueprint.brain_dump}

Provide your genre and style recommendations with explanations."""}
        ]
        
        thinking, content = self.call_llm(messages)
        
        # Parse the response (basic parsing - could be enhanced with structured output)
        lines = content.split('\n')
        for line in lines:
            line_lower = line.lower()
            if line_lower.startswith('genre:'):
                blueprint.genre = line.split(':', 1)[1].strip()
            elif line_lower.startswith('subgenre:'):
                blueprint.subgenre = line.split(':', 1)[1].strip()
            elif line_lower.startswith('tone:'):
                blueprint.tone = line.split(':', 1)[1].strip()
            elif line_lower.startswith('writing style:'):
                blueprint.writing_style = line.split(':', 1)[1].strip()
            elif line_lower.startswith('narrative pov:'):
                blueprint.pov = line.split(':', 1)[1].strip()
        
        blueprint.style_explanation = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="Genre & Style Selection",
            thinking=thinking,
            output=content
        )
        
        return blueprint


class SynopsisAgent(BaseAgent):
    """Agent responsible for creating detailed plot synopsis"""
    
    def __init__(self):
        super().__init__(
            name="Synopsis Agent",
            description="Creates a comprehensive plot synopsis"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a master story architect specializing in plot structure and narrative design. Your role is to create a detailed synopsis for a complete novel.

The synopsis must include:
1. PROTAGONIST - Main character with their defining goal/desire
2. INCITING INCIDENT - The event that starts the story
3. RISING ACTION - Key complications and escalations
4. MAJOR TURNING POINTS - At least 3-4 significant plot turns
5. MIDPOINT - The central pivot of the story
6. CRISIS/DARK MOMENT - The lowest point for the protagonist
7. CLIMAX - The final confrontation/resolution of main conflict
8. RESOLUTION - How the world/character changes after the climax
9. SUBPLOTS - At least 2-3 subplot threads that weave into the main plot
10. THEME STATEMENT - The core thematic message

The synopsis should be detailed enough to guide a full novel (120-200 pages, ~20 chapters).
Write it in present tense, narrative form."""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Crafting detailed plot synopsis")
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Create a detailed synopsis for this novel:

TOPIC: {blueprint.topic}

GENRE: {blueprint.genre} / {blueprint.subgenre}
TONE: {blueprint.tone}
STYLE: {blueprint.writing_style}
POV: {blueprint.pov}

BRAIN DUMP:
{blueprint.brain_dump}

Create a comprehensive synopsis that will guide a 20-chapter novel. Include all major plot points, character arcs, and thematic elements."""}
        ]
        
        thinking, content = self.call_llm(messages)
        blueprint.synopsis = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="Synopsis Creation",
            thinking=thinking,
            output=content
        )
        
        return blueprint


class WorldBuilderAgent(BaseAgent):
    """Agent responsible for creating the world-building bible"""
    
    def __init__(self):
        super().__init__(
            name="World Builder Agent",
            description="Creates comprehensive world-building elements"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a master world-builder specializing in creating rich, detailed fictional universes. Your role is to create a comprehensive world-building bible.

You must create:

## CHARACTERS (5-8 major characters)
For each character provide:
- Name, Age, Role in story
- Personality traits (3-5)
- Strengths (2-3)
- Flaws (2-3)
- Backstory (2-3 sentences)
- Key relationships
- Character arc (how they change)

## LOCATIONS (4-6 key locations)
For each location provide:
- Name and description
- Cultural/social details
- Environmental/atmospheric details
- How it affects the plot

## WORLD SYSTEMS (2-4 systems/rules)
This includes magic systems, technology, politics, religions, economies, etc.
For each system provide:
- Name and description
- Rules/how it works
- Limitations
- How it creates conflict

Be detailed and consistent. Everything should interconnect logically.
Use clear headers and formatting."""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Building comprehensive world bible")
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Create a detailed world-building bible for this novel:

TOPIC: {blueprint.topic}
GENRE: {blueprint.genre} / {blueprint.subgenre}
TONE: {blueprint.tone}

SYNOPSIS:
{blueprint.synopsis}

Create all characters, locations, and world systems. Ensure everything connects to the plot and themes."""}
        ]
        
        thinking, content = self.call_llm(messages)
        
        # Store the raw world-building content
        # In a more advanced version, we'd parse this into structured data
        blueprint.world_building_raw = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="World Building",
            thinking=thinking,
            output=content
        )
        
        return blueprint


class OutlineArchitectAgent(BaseAgent):
    """Agent responsible for creating chapter-by-chapter outline"""
    
    def __init__(self):
        super().__init__(
            name="Outline Architect Agent",
            description="Creates detailed chapter-by-chapter outline"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a master story structure architect specializing in chapter-level plotting. Your role is to create a detailed chapter-by-chapter outline for a complete novel.

For EACH chapter (aim for 18-22 chapters), provide:
1. CHAPTER NUMBER AND TITLE
2. CHAPTER GOAL - What this chapter must accomplish
3. KEY EVENTS - 3-4 major things that happen
4. EMOTIONAL BEATS - The emotional journey of this chapter
5. CONNECTION TO PREVIOUS CHAPTER - How it flows from before
6. SETUP/PAYOFF LINKS - Foreshadowing planted or paid off

Structure requirements:
- Chapters 1-5: Setup and inciting incident
- Chapters 6-10: Rising action and complications
- Chapters 11-14: Midpoint and escalation
- Chapters 15-18: Crisis and build to climax
- Chapters 19-22: Climax and resolution

Each chapter should end with a hook or tension that pulls into the next.
The pacing should escalate toward the climax.

Use clear formatting:
## Chapter [N]: [Title]
**Goal:** ...
**Key Events:**
- ...
**Emotional Beats:** ...
**Connection:** ...
**Foreshadowing:** ..."""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Architecting chapter-by-chapter outline")
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Create a detailed chapter-by-chapter outline:

TOPIC: {blueprint.topic}
GENRE: {blueprint.genre}
TONE: {blueprint.tone}
POV: {blueprint.pov}

SYNOPSIS:
{blueprint.synopsis}

WORLD BUILDING:
{getattr(blueprint, 'world_building_raw', 'See synopsis for details')}

Create 20 chapters with detailed outlines for each. Ensure proper pacing, escalation, and interconnection."""}
        ]
        
        thinking, content = self.call_llm(messages)
        blueprint.outline_raw = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="Chapter Outline",
            thinking=thinking,
            output=content
        )
        
        return blueprint


class SceneChoreographerAgent(BaseAgent):
    """Agent responsible for creating detailed scene beats"""
    
    def __init__(self):
        super().__init__(
            name="Scene Choreographer Agent",
            description="Creates detailed scene beats for each chapter"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a master scene choreographer specializing in breaking chapters into detailed scene beats. Your role is to create a scene-by-scene breakdown.

For each scene within a chapter, provide:
1. SCENE NUMBER + SUMMARY (1 sentence)
2. SETTING - Where and when
3. CHARACTERS PRESENT - Who is in this scene
4. CONFLICT/TENSION - What creates drama
5. OUTCOME - How does the scene end
6. STORY PUSH - How does this move the plot forward

Guidelines:
- Each chapter should have 3-6 scenes
- Vary scene length and intensity
- Include both action and reflection scenes
- Every scene must serve the plot or character development
- End scenes on tension or revelation when possible

Format:
### Scene [N]
**Summary:** ...
**Setting:** ...
**Characters:** ...
**Conflict:** ...
**Outcome:** ...
**Story Push:** ..."""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Choreographing scene beats for all chapters")
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Create detailed scene beats for every chapter:

TOPIC: {blueprint.topic}
GENRE: {blueprint.genre}
POV: {blueprint.pov}

CHAPTER OUTLINE:
{blueprint.outline_raw}

WORLD BUILDING:
{getattr(blueprint, 'world_building_raw', '')}

Create 3-6 scene beats for each chapter. Be specific about what happens in each scene."""}
        ]
        
        thinking, content = self.call_llm(messages)
        blueprint.scene_beats_raw = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="Scene Beats",
            thinking=thinking,
            output=content
        )
        
        return blueprint


class ProseWriterAgent(BaseAgent):
    """Agent responsible for writing the actual prose"""
    
    def __init__(self):
        super().__init__(
            name="Prose Writer Agent",
            description="Writes polished, engaging prose for each chapter"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a master novelist capable of writing engaging, polished prose. Your role is to write complete chapters based on outlines and scene beats.

Writing requirements:
1. RICH DESCRIPTIONS - Paint vivid scenes without purple prose
2. NATURAL DIALOGUE - Characters speak distinctly and realistically
3. VARIED SENTENCE STRUCTURE - Mix short punchy sentences with longer flowing ones
4. SHOW DON'T TELL - Demonstrate emotions through action and dialogue
5. SENSORY DETAILS - Engage all five senses
6. PROPER PACING - Balance action with reflection
7. CHAPTER HOOKS - End with tension or intrigue
8. CONTINUITY - Reference earlier events naturally
9. VOICE CONSISTENCY - Maintain the established style throughout

Chapter format:
## Chapter [N]: [Title]

[Full prose content - aim for 2500-3500 words per chapter]

---

Write in the specified POV and style. Make every sentence count."""

    def write_chapter(self, chapter_num: int, blueprint: NovelBlueprint, 
                      doc_log: DocumentationLog, previous_chapters_summary: str = "") -> str:
        """Write a single chapter"""
        
        # Build context from previous chapters
        context = f"""
NOVEL: {blueprint.topic}
GENRE: {blueprint.genre} / {blueprint.subgenre}
TONE: {blueprint.tone}
WRITING STYLE: {blueprint.writing_style}
POV: {blueprint.pov}

SYNOPSIS:
{blueprint.synopsis}

WORLD BUILDING:
{getattr(blueprint, 'world_building_raw', '')}

FULL OUTLINE:
{blueprint.outline_raw}

SCENE BEATS:
{blueprint.scene_beats_raw}
"""
        
        if previous_chapters_summary:
            context += f"\n\nPREVIOUS CHAPTERS SUMMARY:\n{previous_chapters_summary}"
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""{context}

Now write CHAPTER {chapter_num} in full. 
- Follow the outline and scene beats exactly
- Write 2500-3500 words
- Maintain the established style and voice
- Include callbacks to earlier plot points if this is not chapter 1
- End with a hook that leads into the next chapter

Write the complete chapter now:"""}
        ]
        
        thinking, content = self.call_llm(messages)
        
        doc_log.add_entry(
            agent=self.name,
            stage=f"Chapter {chapter_num} Writing",
            thinking=thinking,
            output=content
        )
        
        return content
    
    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        """Write all chapters"""
        print_agent_start(self.name, "Writing full novel chapters")
        
        # Determine number of chapters from outline (look for "Chapter" patterns)
        import re
        chapter_matches = re.findall(r'Chapter\s+(\d+)', blueprint.outline_raw, re.IGNORECASE)
        if chapter_matches:
            num_chapters = max(int(c) for c in chapter_matches)
        else:
            num_chapters = CHAPTERS_TARGET
        
        print(f"\n{Colors.CYAN}📚 Writing {num_chapters} chapters...{Colors.END}\n")
        
        previous_summary = ""
        
        for i in range(1, num_chapters + 1):
            print_progress(i, num_chapters, "Writing Progress")
            print(f"\n\n{Colors.BOLD}{'═' * 60}{Colors.END}")
            print(f"{Colors.BOLD}Writing Chapter {i} of {num_chapters}{Colors.END}")
            print(f"{Colors.BOLD}{'═' * 60}{Colors.END}\n")
            
            chapter_content = self.write_chapter(i, blueprint, doc_log, previous_summary)
            blueprint.chapters[i] = chapter_content
            
            # Create summary of this chapter for context
            summary_messages = [
                {"role": "system", "content": "You are a concise summarizer. Summarize the key events in 2-3 sentences."},
                {"role": "user", "content": f"Summarize this chapter briefly:\n\n{chapter_content[:3000]}"}
            ]
            
            _, chapter_summary = self.call_llm(summary_messages, stream=False)
            previous_summary += f"\nChapter {i}: {chapter_summary}\n"
            
            # Save progress after each chapter
            save_chapter_to_file(blueprint, i)
        
        return blueprint


class ContinuityGuardianAgent(BaseAgent):
    """Agent responsible for checking and maintaining continuity"""
    
    def __init__(self):
        super().__init__(
            name="Continuity Guardian Agent",
            description="Ensures consistency across all chapters"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a meticulous continuity editor. Your role is to identify and flag any inconsistencies in the novel.

Check for:
1. CHARACTER CONSISTENCY - Names, traits, knowledge
2. TIMELINE ISSUES - Events in wrong order or impossible timing
3. LOCATION CONSISTENCY - Details about places
4. PLOT HOLES - Unresolved setups or contradictions
5. OBJECT TRACKING - Items appearing/disappearing incorrectly
6. RELATIONSHIP CONTINUITY - Character dynamics staying consistent

For each issue found, provide:
- CHAPTER: Where the issue occurs
- ISSUE: What the problem is
- SUGGESTED FIX: How to resolve it

If no issues found, confirm the continuity is solid."""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Checking novel continuity")
        
        # Compile all chapters
        full_novel = "\n\n".join([
            f"CHAPTER {num}:\n{content}" 
            for num, content in sorted(blueprint.chapters.items())
        ])
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Review this complete novel for continuity issues:

WORLD BUILDING REFERENCE:
{getattr(blueprint, 'world_building_raw', '')}

NOVEL TEXT:
{full_novel[:50000]}  # Truncate if too long

Identify any continuity issues and suggest fixes."""}
        ]
        
        thinking, content = self.call_llm(messages)
        blueprint.continuity_notes = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="Continuity Check",
            thinking=thinking,
            output=content
        )
        
        return blueprint


class EditorAgent(BaseAgent):
    """Agent responsible for final polish and refinement"""
    
    def __init__(self):
        super().__init__(
            name="Editor Agent",
            description="Provides final polish and editorial suggestions"
        )
    
    def get_system_prompt(self) -> str:
        return """You are a senior fiction editor with decades of experience. Your role is to provide a final editorial review.

Evaluate:
1. PACING - Does the story move well?
2. CHARACTER ARCS - Are they complete and satisfying?
3. DIALOGUE - Does it sound natural?
4. PROSE QUALITY - Is the writing engaging?
5. OPENING HOOK - Does chapter 1 grab the reader?
6. ENDING SATISFACTION - Is the conclusion earned?
7. THEMATIC COHERENCE - Do themes resonate throughout?

Provide:
- STRENGTHS: What works well
- AREAS FOR IMPROVEMENT: Specific suggestions
- OVERALL ASSESSMENT: Publication readiness

Be constructive and specific."""

    def execute(self, blueprint: NovelBlueprint, doc_log: DocumentationLog) -> NovelBlueprint:
        print_agent_start(self.name, "Final editorial review")
        
        # Get first chapter, middle chapter, and last chapter for review
        chapters = sorted(blueprint.chapters.keys())
        sample_chapters = []
        if chapters:
            sample_chapters.append(("First Chapter", blueprint.chapters[chapters[0]]))
            if len(chapters) > 2:
                mid = chapters[len(chapters)//2]
                sample_chapters.append((f"Middle Chapter ({mid})", blueprint.chapters[mid]))
            sample_chapters.append(("Final Chapter", blueprint.chapters[chapters[-1]]))
        
        sample_text = "\n\n---\n\n".join([
            f"### {name}:\n{content[:3000]}..." 
            for name, content in sample_chapters
        ])
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": f"""Provide editorial review for this novel:

SYNOPSIS:
{blueprint.synopsis}

SAMPLE CHAPTERS:
{sample_text}

Provide your comprehensive editorial assessment."""}
        ]
        
        thinking, content = self.call_llm(messages)
        blueprint.editorial_notes = content
        
        doc_log.add_entry(
            agent=self.name,
            stage="Editorial Review",
            thinking=thinking,
            output=content
        )
        
        return blueprint


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR


def save_chapter_to_file(blueprint: NovelBlueprint, chapter_num: int):
    """Save a single chapter to file"""
    ensure_output_directory()
    filepath = os.path.join(OUTPUT_DIR, f"chapter_{chapter_num:02d}.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(blueprint.chapters.get(chapter_num, ""))


def save_complete_novel(blueprint: NovelBlueprint):
    """Save the complete novel as a single file"""
    ensure_output_directory()
    
    # Create front matter
    front_matter = f"""# {blueprint.topic}

**Genre:** {blueprint.genre} / {blueprint.subgenre}
**Tone:** {blueprint.tone}
**Style:** {blueprint.writing_style}

---

## Synopsis

{blueprint.synopsis}

---

"""
    
    # Compile all chapters
    chapters_text = "\n\n---\n\n".join([
        blueprint.chapters[num] 
        for num in sorted(blueprint.chapters.keys())
    ])
    
    full_novel = front_matter + chapters_text
    
    filepath = os.path.join(OUTPUT_DIR, "complete_novel.md")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_novel)
    
    print(f"\n{Colors.GREEN}✅ Complete novel saved to: {filepath}{Colors.END}")
    return filepath


def save_documentation(doc_log: DocumentationLog, blueprint: NovelBlueprint):
    """Save all documentation and process logs"""
    ensure_output_directory()
    
    # Save documentation log
    doc_log.save(os.path.join(OUTPUT_DIR, "documentation_log.json"))
    
    # Save blueprint stages
    stages = {
        "topic": blueprint.topic,
        "brain_dump": blueprint.brain_dump,
        "genre": blueprint.genre,
        "subgenre": blueprint.subgenre,
        "tone": blueprint.tone,
        "writing_style": blueprint.writing_style,
        "pov": blueprint.pov,
        "style_explanation": blueprint.style_explanation,
        "synopsis": blueprint.synopsis,
        "world_building": getattr(blueprint, 'world_building_raw', ''),
        "outline": getattr(blueprint, 'outline_raw', ''),
        "scene_beats": getattr(blueprint, 'scene_beats_raw', ''),
        "continuity_notes": getattr(blueprint, 'continuity_notes', ''),
        "editorial_notes": getattr(blueprint, 'editorial_notes', '')
    }
    
    with open(os.path.join(OUTPUT_DIR, "novel_blueprint.json"), 'w', encoding='utf-8') as f:
        json.dump(stages, f, indent=2, ensure_ascii=False)
    
    # Save a human-readable development document
    dev_doc = f"""# Novel Development Document

## Original Topic
{blueprint.topic}

## Brain Dump
{blueprint.brain_dump}

## Genre & Style Analysis
{blueprint.style_explanation}

## Synopsis
{blueprint.synopsis}

## World Building Bible
{getattr(blueprint, 'world_building_raw', 'Not generated')}

## Chapter Outline
{getattr(blueprint, 'outline_raw', 'Not generated')}

## Scene Beats
{getattr(blueprint, 'scene_beats_raw', 'Not generated')}

## Continuity Notes
{getattr(blueprint, 'continuity_notes', 'Not generated')}

## Editorial Assessment
{getattr(blueprint, 'editorial_notes', 'Not generated')}
"""
    
    with open(os.path.join(OUTPUT_DIR, "development_document.md"), 'w', encoding='utf-8') as f:
        f.write(dev_doc)
    
    print(f"{Colors.GREEN}✅ Documentation saved to: {OUTPUT_DIR}/{Colors.END}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class NovelOrchestrator:
    """Main orchestrator that coordinates all agents"""
    
    def __init__(self):
        self.agents = [
            BrainDumpAgent(),
            GenreStyleAgent(),
            SynopsisAgent(),
            WorldBuilderAgent(),
            OutlineArchitectAgent(),
            SceneChoreographerAgent(),
            ProseWriterAgent(),
            ContinuityGuardianAgent(),
            EditorAgent()
        ]
        self.doc_log = DocumentationLog()
    
    def generate_novel(self, topic: str) -> NovelBlueprint:
        """Execute the full novel generation pipeline"""
        
        print_banner("MULTI-AGENT NOVEL GENERATION SYSTEM", "═", 80)
        print(f"{Colors.BOLD}Topic:{Colors.END} {topic}\n")
        print(f"{Colors.DIM}Model: {MODEL_NAME}")
        print(f"Thinking Mode: {'Enabled' if ENABLE_THINKING else 'Disabled'}")
        print(f"Output Directory: {OUTPUT_DIR}{Colors.END}\n")
        
        # Initialize blueprint
        blueprint = NovelBlueprint(topic=topic)
        
        start_time = time.time()
        
        # Execute each agent in sequence
        for i, agent in enumerate(self.agents, 1):
            print_section(f"Stage {i}/{len(self.agents)}: {agent.name}")
            print(f"{Colors.DIM}{agent.description}{Colors.END}\n")
            
            try:
                blueprint = agent.execute(blueprint, self.doc_log)
                print(f"\n{Colors.GREEN}✓ {agent.name} completed successfully{Colors.END}")
            except Exception as e:
                print(f"\n{Colors.RED}✗ {agent.name} failed: {str(e)}{Colors.END}")
                self.doc_log.add_entry(
                    agent=agent.name,
                    stage="Error",
                    thinking="",
                    output=f"Error: {str(e)}"
                )
        
        elapsed_time = time.time() - start_time
        
        # Save everything
        print_section("Saving Outputs")
        save_complete_novel(blueprint)
        save_documentation(self.doc_log, blueprint)
        
        # Print summary
        print_banner("GENERATION COMPLETE", "═", 80)
        
        word_count = sum(len(ch.split()) for ch in blueprint.chapters.values())
        chapter_count = len(blueprint.chapters)
        
        print(f"""
{Colors.BOLD}Summary:{Colors.END}
  • Chapters Generated: {chapter_count}
  • Total Word Count: {word_count:,}
  • Estimated Pages: {word_count // 250}
  • Generation Time: {elapsed_time/60:.1f} minutes
  
{Colors.BOLD}Output Files:{Colors.END}
  • {OUTPUT_DIR}/complete_novel.md
  • {OUTPUT_DIR}/development_document.md
  • {OUTPUT_DIR}/documentation_log.json
  • {OUTPUT_DIR}/novel_blueprint.json
  • {OUTPUT_DIR}/chapter_XX.md (individual chapters)
""")
        
        return blueprint


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    
    print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███╗   ██╗ ██████╗ ██╗   ██╗███████╗██╗         ██████╗ ███████╗███╗   ██╗ ║
║   ████╗  ██║██╔═══██╗██║   ██║██╔════╝██║        ██╔════╝ ██╔════╝████╗  ██║ ║
║   ██╔██╗ ██║██║   ██║██║   ██║█████╗  ██║        ██║  ███╗█████╗  ██╔██╗ ██║ ║
║   ██║╚██╗██║██║   ██║╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║██╔══╝  ██║╚██╗██║ ║
║   ██║ ╚████║╚██████╔╝ ╚████╔╝ ███████╗███████╗   ╚██████╔╝███████╗██║ ╚████║ ║
║   ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚══════╝╚══════╝    ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ║
║                                                                              ║
║              Multi-Agent Novel Generation System using Ollama                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}
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
    
    # Create orchestrator and generate novel
    orchestrator = NovelOrchestrator()
    
    try:
        blueprint = orchestrator.generate_novel(topic)
        print(f"\n{Colors.GREEN}🎉 Novel generation complete!{Colors.END}\n")
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Generation interrupted by user.{Colors.END}")
        print(f"{Colors.DIM}Partial output may be available in {OUTPUT_DIR}/{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error during generation: {str(e)}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
