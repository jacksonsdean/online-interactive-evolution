import json

print('Loading function')


def lambda_handler(event, context):
    ids = event["queryStringParameters"]['ids']
    return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': json.dumps(str(ids))
        }