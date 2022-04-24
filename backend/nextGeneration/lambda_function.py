"""Handler for Next Generation."""
import json

def lambda_handler(event, context):
    """Handle an incoming request from Next Generation."""
    body = None
    status = 200
    try:
        print("event:", event)
        body_in = json.loads(event['body'])
        ids = body_in['ids']
        print("context:", context)
        body = json.dumps(str(ids))
    except Exception as e:
        print(type(e), e)
        status = 500
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
