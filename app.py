
import aws_cdk as cdk
from new_ai.new_ai_stack import NewAiStack

app = cdk.App()
NewAiStack(app, "NewAiStack")
app.synth()
