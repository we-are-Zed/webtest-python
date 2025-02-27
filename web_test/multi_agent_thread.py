import logging
import os.path
import threading
from datetime import datetime
from typing import Tuple, Optional, List, Dict
from urllib.parse import urlparse

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from urllib3.exceptions import ReadTimeoutError

from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from action.web_action_detector import WebActionDetector
from config.log_config import LogConfig
from config.settings import settings
from exceptions import NoActionsException
from multi_agent.multi_agent_system import MultiAgentSystem
from state.impl.action_execute_failed_state import ActionExecuteFailedState
from state.impl.out_of_domain_state import OutOfDomainState
from state.impl.same_url_state import SameUrlState
from state.web_state import WebState
from utils import instantiate_class_by_module_and_class_name, get_class_by_module_and_class_name

logger = logging.getLogger(__name__)
logger.addHandler(LogConfig.get_file_handler())


class MultiAgentThread(threading.Thread):
    def __init__(self, chrome_options: Options, agent_name: str, multi_agent_system: MultiAgentSystem) -> None:
        super().__init__()
        self.driver = webdriver.Chrome(chrome_options, Service(executable_path=settings.driver_path))
        self.agent_name = agent_name
        self.multi_agent_system = multi_agent_system
        logger.info("Thread " + self.agent_name + ":  " + "Webdriver created successfully")
        self.chrome_options = chrome_options
        self.action_detector_class: type = get_class_by_module_and_class_name(settings.action_detector["module"],
                                                                              settings.action_detector["class"])
        if settings.action_detector["class"] == "CombinationDetector":
            detectors: List[WebActionDetector] = []
            for detector_name in settings.action_detector["detectors"]:
                detectors.append(
                    instantiate_class_by_module_and_class_name(detector_name["module"], detector_name["class"]))
            self.action_detector: WebActionDetector = self.action_detector_class(detectors)
        else:
            self.action_detector: WebActionDetector = self.action_detector_class()
        self.state_class: type = get_class_by_module_and_class_name(settings.state["module"], settings.state["class"])

        self.prev_state: Optional[WebState] = None
        self.current_state: Optional[WebState] = None
        self.action_dict: Dict[WebAction, int] = {}
        self.state_dict: Dict[WebState, int] = {}
        self.transition_record_list: List[Tuple[Optional[WebState], WebAction, WebState]] = []
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.screen_shot_count = 0
        self.screen_shot_record_round = 50
        self.same_url_count = 0
        self.current_url = "111"
        self.check_result = False
        self.restart_interval = settings.restart_interval

    def run(self):
        logger.info("Thread " + self.agent_name + ":  " + "Execution start")
        self.driver.get(settings.entry_url)
        self.stop_event.wait(10)
        html = self.init_state()
        self.driver.set_page_load_timeout(settings.page_load_timeout)
        continuous_restart_count = 0
        while not self.stop_event.is_set():
            chosen_action = None
            try:
                chosen_action = self.multi_agent_system.get_action(self.current_state, html, self.agent_name,
                                                                   self.current_url, self.check_result)
                logger.info("Thread " + self.agent_name + ":  " + f" Chosen action: {chosen_action}")
                if not isinstance(chosen_action, RestartAction):
                    self.action_dict[chosen_action] += 1
                    continuous_restart_count = 0
                else:
                    continuous_restart_count += 1
                if continuous_restart_count >= settings.continuous_restart_threshold:
                    self.restart_webdriver()
                    continuous_restart_count = 0
                chosen_action.execute(self.driver)
                WebDriverWait(self.driver, settings.page_load_timeout).until(
                    lambda x: x.execute_script('return document.readyState') == 'complete'
                )
                self.stop_event.wait(2)
                check_result, domain = self.check_domain()
                self.check_result = check_result
                if check_result and self.same_url_count < self.restart_interval:
                    self.trace_error()
                    action_list = self.action_detector.get_actions(self.driver)
                    with self.lock:
                        for action in action_list:
                            self.action_dict.setdefault(action, 0)
                    if self.driver.current_url == self.current_url:
                        self.same_url_count += 1
                    else:
                        self.same_url_count = 0
                        self.current_url = self.driver.current_url
                    new_state = self.multi_agent_system.get_state(
                        self.state_class(action_list, self.driver.current_url))
                    html = self.driver.page_source
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.info("Thread " + self.agent_name + ":  " + f"Execute action success, New state: {new_state}")
                elif not check_result:
                    restart_url = self.multi_agent_system.get_restart_url(self.agent_name)
                    new_state = OutOfDomainState(restart_url)
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.warning("Thread " + self.agent_name + ":  " + f"Out of domain, New state: {new_state}")
                else:
                    self.same_url_count = 0
                    restart_url = self.multi_agent_system.get_restart_url(self.agent_name)
                    new_state = SameUrlState(restart_url)
                    self.transit(chosen_action, new_state)
                    if settings.enable_screen_shot:
                        self.save_screen_shot()
                    logger.warning(
                        "Thread " + self.agent_name + ":  " + f"Same url too many times, New state: {new_state}")
            except (NoActionsException, WebDriverException,ReadTimeoutError, TimeoutError) as e:
                restart_url = self.multi_agent_system.get_restart_url(self.agent_name)
                new_state = ActionExecuteFailedState(restart_url)
                self.transit(chosen_action, new_state)
                # 如果重启仍然失败，多增加一些计数，避免多次在同一个url重启
                if chosen_action is not None and isinstance(chosen_action, RestartAction):
                    self.multi_agent_system.restart_fail(agent_name=self.agent_name, restart_url=restart_url)
                if isinstance(e, NoActionsException):
                    logger.warning("Thread " + self.agent_name + ":  " +
                                   f"Choose action failed, no actions, New state: {new_state}")
                else:
                    logger.warning("Thread " + self.agent_name + ":  " +
                                   f"Execute action failed ({type(e).__name__}), New state: {new_state}")
            except (Exception, KeyboardInterrupt) as e:
                logger.exception(f"Thread {self.agent_name}: Error happened")
                self.restart_webdriver()
                continuous_restart_count = 0
                self.driver.get(settings.entry_url)
                self.stop_event.wait(2)
                html = self.init_state()
                self.multi_agent_system.deal_exception(self.agent_name)
            if len(self.driver.window_handles) > 1:
                self.driver.switch_to.window(self.driver.window_handles[1])
                self.close_other_windows()
        self.driver.quit()

    def init_state(self):
        action_list = self.action_detector.get_actions(self.driver)
        self.current_state = self.multi_agent_system.get_state(
            self.state_class(action_list, self.driver.current_url))
        with self.lock:
            for action in action_list:
                self.action_dict.setdefault(action, 0)
            self.state_dict[self.current_state] = 1
        html: str = self.driver.page_source
        self.trace_error()
        logger.info("Thread " + self.agent_name + ":  " + f"Initial state: {self.current_state}")
        return html

    def trace_error(self):
        with self.lock:
            logs = self.driver.get_log("browser")
            with open(os.path.join(settings.output_path, "bug.log"), "a", encoding="utf-8") as f:
                for log in logs:
                    if (log["level"] == "WARNING") or (log["level"] == "SEVERE"):
                        logger.info("Thread " + self.agent_name + ":  " + f"Detect browser error: {log}")
                        f.write(str(log) + "\n")

    def transit(self, chosen_action: WebAction, new_state: WebState) -> None:
        if chosen_action is not None:
            self.current_state.update_action_execution_time(chosen_action)
            self.current_state.update_transition_information(chosen_action, new_state)
        self.prev_state = self.current_state
        self.current_state = new_state

    def check_domain(self) -> Tuple[bool, str]:
        current_url = self.driver.current_url
        domain = urlparse(current_url).netloc.lower()
        if domain in settings.domains:
            return True, domain
        else:
            return False, domain

    def close_other_windows(self) -> None:
        current_window = self.driver.current_window_handle
        for handle in self.driver.window_handles:
            if handle != current_window:
                self.driver.switch_to.window(handle)
                self.driver.close()
        self.driver.switch_to.window(current_window)

    def restart_webdriver(self) -> None:
        self.driver.quit()
        self.driver = webdriver.Chrome(self.chrome_options, Service(executable_path=settings.driver_path))
        self.driver.set_page_load_timeout(settings.page_load_timeout)
        logger.info("Thread " + self.agent_name + ":  " + "Webdriver restart successfully")

    def stop(self):
        self.stop_event.set()

    def save_screen_shot(self):
        if self.screen_shot_count % self.screen_shot_record_round == 0:
            # Save the original window size
            original_window_size = self.driver.get_window_size()

            # Execute JavaScript to get the full page height
            js = "return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);"
            scroll_height = self.driver.execute_script(js)

            # Set the window size to the full page height and take the screenshot
            self.driver.set_window_size(original_window_size['width'], scroll_height)
            screenshot = self.driver.get_screenshot_as_png()

            # Reset the window size to the original size
            self.driver.set_window_size(original_window_size['width'], original_window_size['height'])

            folder_path = os.path.join(settings.output_path, "ScreenShots/agent-" + self.agent_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path,
                                     "screenShot-" + f"{self.screen_shot_count}" + "---" + datetime.now().strftime(
                                         "%Y-%m-%d_%H_%M_%S") + ".png")
            with open(file_path, "wb") as file:
                file.write(screenshot)
        self.screen_shot_count += 1
