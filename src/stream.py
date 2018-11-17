import requests
import pandas as pd
api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'
url = 'https://hxobin8em5.execute-api.us-west-2.amazonaws.com/api/'


def get_stream(sequence):
    sequence_number = sequence
    response = requests.post(url, json={'api_key': api_key,
                                        'sequence_number': sequence_number})
    raw_data = response.json()

        
    return raw_data


def get_stream_dfs(count):
    """get several items from a stream of data from ec2 instance"""
    stream_df = pd.DataFrame()
    for i in range(count):
        raw = get_stream(i)
        df = pd.DataFrame(raw['data'])
        stream_df = pd.concat([stream_df, df], axis=0)
    return stream_df
