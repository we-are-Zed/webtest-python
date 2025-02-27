import ast
import json
import os


def get_state(session):
    state_result = set()
    for state in session['state_list']:
        info = state.get('info', '')
        if 'ActionSetWithExecutionTimesState' in info:
            state_result.add(info)
    return len(state_result)



def get_action(session):
    action_result = set()
    for action in session['action_list']:
        action_result.add(action[0])
    return len(action_result)


def get_visited_action(session):
    visited_action_num = 0
    for action in session['action_list']:
        if action[1] > 0:
            visited_action_num += 1
    return visited_action_num


def get_execution_times(session):
    execution_times = 0
    for action in session['action_list']:
        if action[1] > 0:
            execution_times += action[1]
    return execution_times


def get_bugs(bug_path):
    bug_set = set()
    with open(bug_path, 'r', encoding="utf-8") as file:
        for line in file:
            try:
                # 将每一行解析为字典
                log_entry = ast.literal_eval(line.strip())
                if not isinstance(log_entry, dict) or 'message' not in log_entry:
                    continue
                message = log_entry['message'].lower()
                bug_set.add(message)

            except Exception as e:
                pass
                # print(f"Error parsing line: {line.strip()} - {e}")

    bug_count = len(bug_set)
    return bug_count


def get_bugs_filter(bug_path):
    bug_set = set()
    with open(bug_path, 'r', encoding="utf-8") as file:
        for line in file:
            try:
                # 将每一行解析为字典
                log_entry = ast.literal_eval(line.strip())
                if not isinstance(log_entry, dict) or 'message' not in log_entry:
                    continue

                # 提取 message 信息
                message = log_entry['message'].lower()

                # 过滤条件
                if 'level' in log_entry and log_entry['level'].lower() == 'severe':
                    continue  # 跳过 SEVERE 级别的错误
                if any(keyword in message for keyword in ['bug', 'error', 'failed']):
                    bug_set.add(message)

            except Exception as e:
                pass
                # print(f"Error parsing line: {line.strip()} - {e}")

    bug_count = len(bug_set)
    return bug_count


if __name__ == '__main__':
    mas = 'D:\Projects\webtest-python\webtest_output/result-2'
    for item in os.scandir(mas):
        if item.is_dir():
            session_path = os.path.join(item.path, 'output_data/newest.json')
            session = json.load(open(session_path, 'r'))
            bug_path = os.path.join(item.path, 'bug.log')

            print('-----------------------------------------')
            print('Config name: ', item.name)
            print('State: ', get_state(session))
            # print('State/e: ', get_state(session) * 1000.0 / get_execution_times(session))
            print('Action: ', get_action(session))
            print('Visited: ', get_visited_action(session))
            print('Bugs: ', get_bugs(bug_path))
            print('Bugs filtered: ', get_bugs_filter(bug_path))
            print('Execution times: ', get_execution_times(session))
