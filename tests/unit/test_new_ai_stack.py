import aws_cdk as core
import aws_cdk.assertions as assertions

from new_ai.new_ai_stack import NewAiStack

# example tests. To run these tests, uncomment this file along with the example
# resource in new_ai/new_ai_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = NewAiStack(app, "new-ai")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
