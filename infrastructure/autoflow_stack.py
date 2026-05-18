"""
AutoFlow AI — AWS CDK Infrastructure Stack
Defines Lambda + API Gateway deployment for the anomaly detection service.
Deploy locally with LocalStack via: cdklocal deploy
Deploy to real AWS via: cdk deploy
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as _lambda,
    aws_apigateway as apigw,
    aws_logs as logs,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct


class AutoflowStack(Stack):
    """
    AutoFlow AI infrastructure stack.

    Resources created:
    - Lambda function running the FastAPI app (via Mangum ASGI adapter)
    - API Gateway REST API proxying all requests to Lambda
    - CloudWatch log group for Lambda logs
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ── CloudWatch Log Group ──────────────────────────────────────────
        log_group = logs.LogGroup(
            self,
            "AutoflowLogs",
            log_group_name="/autoflow-ai/lambda",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ── Lambda Function ───────────────────────────────────────────────
        # Packages app/ + requirements into a Lambda deployment bundle.
        # Mangum translates API Gateway events into ASGI requests for FastAPI.
        autoflow_fn = _lambda.Function(
            self,
            "AutoflowFunction",
            function_name="autoflow-ai",
            runtime=_lambda.Runtime.PYTHON_3_11,
            handler="app.main.handler",           # Mangum handler
            code=_lambda.Code.from_asset(
                "..",                             # project root
                exclude=[
                    "infrastructure",
                    "tests",
                    ".git",
                    "__pycache__",
                    "*.pyc",
                    ".env*",
                ],
            ),
            timeout=Duration.seconds(30),
            memory_size=512,
            environment={
                "PYTHONUNBUFFERED": "1",
            },
            log_group=log_group,
        )

        # ── API Gateway ───────────────────────────────────────────────────
        api = apigw.RestApi(
            self,
            "AutoflowApi",
            rest_api_name="autoflow-ai-api",
            description="AutoFlow AI — Vehicle Sensor Anomaly Detection API",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=100,
                throttling_burst_limit=200,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        # Proxy ALL requests to Lambda (FastAPI handles routing internally)
        lambda_integration = apigw.LambdaIntegration(
            autoflow_fn,
            proxy=True,
            allow_test_invoke=True,
        )
        api.root.add_proxy(
            default_integration=lambda_integration,
            any_method=True,
        )

        # ── Stack Outputs ─────────────────────────────────────────────────
        CfnOutput(
            self,
            "ApiUrl",
            value=api.url,
            description="AutoFlow AI API Gateway URL",
        )
        CfnOutput(
            self,
            "LambdaFunctionName",
            value=autoflow_fn.function_name,
            description="Lambda function name",
        )
        CfnOutput(
            self,
            "DocsUrl",
            value=f"{api.url}docs",
            description="FastAPI interactive docs",
        )
