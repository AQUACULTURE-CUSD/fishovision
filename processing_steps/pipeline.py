from abc import abstractmethod, ABC


class ProcessingStep(ABC):
    """
    Inherit from the class, such as MyClass(ProcessingStep) and then override the process(self, context)
    method to be able to add your class to the pipeline.
    """
    @abstractmethod
    def process(self, context: dict) -> dict:
        """
        Processes data from the context dictionary and returns the updated context.
        'context' is a dictionary used to pass data between steps.
        For example: may use context['current_image'] to get the current image,
        and can store arbitrary items in the context for later use (not just the modified image).
        """
        pass


class Pipeline:
    """
    The holder for the pipeline steps. Runs all the steps in the pipeline on a particular input
    context. The user should create a dictionary that contains the input context items, and then
    pass it into the pipeline.run(context) method. Each input context should correspond to one
    frame/image, and then the pipeline will run the entire pipeline on the one image (before
    continuing to the next image).
    """
    def __init__(self, steps: list[ProcessingStep]):
        """
        Saves the input list of processing steps.
        """
        self.steps = steps

    def run(self, context: dict) -> dict:
        """
        Runs the data through all registered steps.
        """
        for step in self.steps:
            context = step.process(context)
        return context
