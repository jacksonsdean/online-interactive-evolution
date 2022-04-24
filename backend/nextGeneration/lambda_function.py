"""Handler for Next Generation."""
import json

def lambda_handler(event, context):
    """Handle an incoming request from Next Generation."""
    body = None
    status = 200
    try:
        ids = event['ids']
        print("context:", context)
        body = json.dumps(str(ids))
    except AttributeError as e:
        print(e)
        body = json.dumps(f"{type(e)}: {e}")

    return {
            'statusCode': status,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': body
        }
