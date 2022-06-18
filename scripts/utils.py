import urllib.request as req

import json
import urllib
import sys

def write_json_file(dic, target_file):
    json_object = json.dumps(dic, indent = 4)
    with open(target_file, "w") as outfile:
        outfile.write(json_object)

def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)

def download_file_from_url(url, target):
    try:
        req.urlretrieve(url, target)
    except urllib.error.HTTPError as err:
        # print('========================')
        # print(f'Error Code: {err.getcode()}')
        # print(f'Target URL: {url}' )
        # print('========================')
        # print('\n')
        pass
    except urllib.error.URLError as err:
        # print('========================')
        # print(f'Error Code: WinError 10060 - Connection Problem')
        # print(f'Target URL: {url}' )
        # print('========================')
        # print('\n')
        pass
    except Exception as e:
        # print('========================')
        # print(f'Error Code: Unknown Error')
        # print(f'Target URL: {url}' )
        # print('========================')
        # print('\n')
        pass
    except KeyboardInterrupt:
        sys.exit()
        pass
    return True