from duckduckgo_search import DDGS
from griptape.artifacts import TextArtifact
from griptape.drivers import LocalStructureRunDriver
from griptape.rules import Rule
from griptape.structures import Agent, Pipeline, Workflow
from griptape.tasks import CodeExecutionTask, PromptTask, StructureRunTask
from griptape.tools import StructureRunClient, TaskMemoryClient, WebScraper

# 1. Create Tool

def search_duckduckgo(task: CodeExecutionTask) -> TextArtifact:
    keywords = task.input.value
    results = DDGS().text(keywords, max_results=5)
    return TextArtifact(results)

def build_search_pipeline() -> Pipeline:
    pipeline = Pipeline()
    pipeline.add_task(
        CodeExecutionTask(
            "{{ args[0] }}",
            run_fn=search_duckduckgo,
        ),
    )
    return pipeline

search_driver = LocalStructureRunDriver(structure_factory_fn=build_search_pipeline)
search_tool = StructureRunClient(
    name="Search tool",
    description="Search the web for information",
    driver=search_driver,
    off_prompt=True,
)

# 2. Create Agents

def build_researcher():
    """Builds a Researcher Structure."""
    researcher = Agent(
        id="researcher",
        tools=[
            search_tool,
            WebScraper(
                off_prompt=True,
            ),
            TaskMemoryClient(off_prompt=False),
        ],
        rules=[
            Rule("Position: Lead Research Analyst"),
            Rule(
                "Objective: Discover innovative advancements in artificial intelligence and data analytics."
            ),
            Rule(
                "Background: You are part of a prominent technology research institute. Your speciality is spotting new trends. You excel at analyzing intricate data and delivering practical insights."
            ),
        ],
    )
    return researcher

def build_writer(role: str, goal: str, backstory: str):
    """Builds a Writer Structure.

    Args:
        role: The role of the writer.
        goal: The goal of the writer.
        backstory: The backstory of the writer.
    """
    writer = Agent(
        id=role.lower().replace(" ", "_"),
        rules=[
            Rule(f"Position: {role}"),
            Rule(f"Objective: {goal}"),
            Rule(f"Backstory: {backstory}"),
            Rule("Desired Outcome: Full blog post of at least 4 paragraphs"),
        ],
    )
    return writer


if __name__ == "__main__":
    team = Workflow()
    
    # 3. Create Tasks
    research_task = team.add_task(
        StructureRunTask(
            (
                """Perform a detailed examination of the newest developments in AI as of 2024.
                Pinpoint major trends, breakthroughs, and their implications for various industries.""",
            ),
            id="research",
            driver=LocalStructureRunDriver(
                structure_factory_fn=build_researcher,
            ),
        ),
    )
    
    WRITERS = [
        {
            "role": "Travel Adventure Blogger",
            "goal": "Inspire wanderlust with stories of hidden gems and exotic locales",
            "backstory": "With a passport full of stamps, you bring distant cultures and breathtaking scenes to life through vivid storytelling and personal anecdotes.",
        },
        {
            "role": "Lifestyle Freelance Writer",
            "goal": "Share practical advice on living a balanced and stylish life",
            "backstory": "From the latest trends in home decor to tips for wellness, your articles help readers create a life that feels both aspirational and attainable.",
        },
    ]
    
    team_tasks = []
    for writer in WRITERS:
        team_tasks.append(
            StructureRunTask(
                (
                    """Using insights provided, develop an engaging blog
                post that highlights the most significant AI advancements.
                Your post should be informative yet accessible, catering to a tech-savvy audience.
                Make it sound cool, avoid complex words so it doesn't sound like AI.

                Insights:
                {{ parent_outputs["research"] }}""",
                ),
                driver=LocalStructureRunDriver(
                    structure_factory_fn=lambda: build_writer(
                        role=writer["role"],
                        goal=writer["goal"],
                        backstory=writer["backstory"],
                    )
                ),
            )
        )

    end_task = team.add_task(
        PromptTask(
            "State: All Done!",
        )
    )
    
    team.insert_tasks(research_task, team_tasks, end_task)
    team.run()
