#!/usr/bin/env python3
"""
AutoFlow AI — CDK App Entry Point
Run: cdklocal deploy  (LocalStack)
Run: cdk deploy       (real AWS)
"""
import aws_cdk as cdk
from autoflow_stack import AutoflowStack

app = cdk.App()

AutoflowStack(
    app,
    "AutoflowAiStack",
    description="AutoFlow AI — Vehicle Sensor Anomaly Detection on AWS Lambda + API Gateway",
)

app.synth()
