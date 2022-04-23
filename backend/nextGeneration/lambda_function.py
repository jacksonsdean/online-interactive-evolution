"""Handler for Next Generation."""
import json

def lambda_handler(event, context):
    """Handle an incoming request from Next Generation."""
    try:
        ids = event['ids']
        print("context:", context)
        return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
                },
                'body': json.dumps(str(ids))
            }
    except AttributeError as e:
        print(e)
        return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
                },
                'body': json.dumps(f"{type(e)}: {e}")
            }
