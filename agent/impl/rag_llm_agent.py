import os
import random
import re
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any
import math

from action.impl.click_action import ClickAction
from action.impl.random_input_action import RandomInputAction
from action.impl.random_select_action import RandomSelectAction
from action.impl.restart_action import RestartAction
from action.web_action import WebAction
from agent.agent import Agent
from state.web_state import WebState
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 处理Chroma版本兼容性
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

from langchain.embeddings.base import Embeddings
from langchain.schema import Document


def manage_vectorstore(vectorstore, max_entries=None, cleanup_ratio=0.8, kb_name="KB", 
                      close_connection=False, verbose=False):
    """
    🚀 统一的向量存储管理函数 - 合并清理和连接管理功能
    
    Args:
        vectorstore: 向量存储对象
        max_entries: 最大记录数，None表示不清理记录
        cleanup_ratio: 清理后保留的比例（0.8 = 保留80%）
        kb_name: 知识库名称（用于日志）
        close_connection: 是否关闭连接
        verbose: 是否显示详细信息
    
    Returns:
        bool: 是否执行了清理操作
    """
    if vectorstore is None:
        return False
    
    cleaned = False
    
    # 1. 记录清理功能
    if max_entries is not None:
        try:
            if hasattr(vectorstore, '_collection'):
                current_count = vectorstore._collection.count()
                if current_count > max_entries:
                    if verbose:
                        print(f"🧹 {kb_name} cleanup: {current_count} > {max_entries}")
                    
                    # 计算删除数量
                    target_count = int(max_entries * cleanup_ratio)
                    excess_count = current_count - target_count
                    
                    try:
                        # 获取所有记录的ID和时间戳
                        all_data = vectorstore._collection.get(include=['metadatas'])
                        
                        # 按时间戳排序
                        records_with_ids = []
                        for i, metadata in enumerate(all_data['metadatas']):
                            timestamp = metadata.get('timestamp', '1970-01-01T00:00:00')
                            records_with_ids.append((timestamp, all_data['ids'][i]))
                        
                        # 删除最旧的记录
                        records_with_ids.sort(key=lambda x: x[0])
                        ids_to_delete = [record[1] for record in records_with_ids[:excess_count]]
                        
                        if ids_to_delete:
                            vectorstore._collection.delete(ids=ids_to_delete)
                            if verbose:
                                print(f"🧹 {kb_name}: Deleted {len(ids_to_delete)} records (kept {target_count})")
                            cleaned = True
                    
                    except Exception as delete_error:
                        if verbose:
                            print(f"⚠️ {kb_name}: Delete failed: {delete_error}")
                
                elif verbose and current_count > max_entries * 0.8:
                    print(f"📊 {kb_name}: {current_count} records (limit: {max_entries})")
                    
        except Exception as e:
            if verbose:
                print(f"Error checking {kb_name} records: {e}")
    
    # 2. 连接关闭功能
    if close_connection:
        try:
            if verbose:
                print(f"🔌 Closing {kb_name} connections...")
            
            # 关闭Chroma客户端
            if hasattr(vectorstore, '_client') and vectorstore._client:
                vectorstore._client.reset()
            
            # 清理集合引用
            if hasattr(vectorstore, '_collection'):
                del vectorstore._collection
            
        except Exception as e:
            if verbose:
                print(f"Error closing {kb_name} connections: {e}")
    
    return cleaned


# 与大型语言模型（LLM）交互 - 使用SiliconFlow API
class LLMInterface:
    def __init__(self, params, verbose=False):
        """
        使用SiliconFlow API进行LLM交互
        """
        self.api_key = params.get("api_key", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.chat_url = params.get("chat_url", "https://api.siliconflow.cn/v1/chat/completions")
        self.model = params.get("model", "deepseek-ai/DeepSeek-R1")
        self.enable_thinking = params.get("enable_thinking", False)
        self.max_tokens = params.get("max_tokens", 1024)
        self.temperature = params.get("temperature", 0.7)
        self.verbose = verbose  # 添加 verbose 属性
        
        # 为每个LLM实例生成唯一会话标识符
        import uuid
        self.session_id = str(uuid.uuid4())
        self.test_session_counter = 0  # 测试会话计数器
        
        if self.verbose:
            print(f"LLM会话ID: {self.session_id}")

    def reset_session(self):
        """
        重置会话状态 - 为新测试创建完全独立的会话
        """
        import uuid
        old_session_id = self.session_id
        self.session_id = str(uuid.uuid4())
        self.test_session_counter += 1
        
        if self.verbose:
            print(f"LLM会话重置:")
            print(f"  旧会话ID: {old_session_id}")
            print(f"  新会话ID: {self.session_id}")
            print(f"  测试计数: {self.test_session_counter}")

    def _build_isolation_prompt(self, user_prompt: str) -> str:
        """
        构建包含隔离信息的提示词，确保测试独立性
        """
        isolation_header = f"""
[测试隔离声明]
- 这是一个全新的独立测试会话
- 会话ID: {self.session_id}
- 测试编号: {self.test_session_counter}
- 请忽略任何之前的对话历史和记忆
- 请基于当前提供的信息独立做出决策
- 不要参考之前测试的结果或经验

[当前测试任务]
{user_prompt}

请严格基于上述信息进行推理和决策，确保测试的独立性和一致性。
"""
        return isolation_header.strip()
        
    def chat_with_thinking(self, prompt: str) -> Dict[str, str]:
        """
        发送聊天请求并返回内容和thinking过程
        返回: {"content": "回答内容", "reasoning": "思考过程"}
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # 添加会话标识符到请求头
            "X-Session-ID": self.session_id,
            "X-Test-Session": str(self.test_session_counter)
        }
        
        # 构建包含隔离信息的提示词
        isolated_prompt = self._build_isolation_prompt(prompt)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"你是一个Web测试专家。当前会话ID: {self.session_id}。这是一个独立的测试会话，请忽略任何之前的对话记忆。"
                },
                {
                    "role": "user", 
                    "content": isolated_prompt
                }
            ],
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            # 添加随机种子确保每次调用的独立性
            "seed": hash(self.session_id + str(self.test_session_counter)) % 2147483647
        }
        
        # 只有支持thinking的模型才添加相关参数
        if "QwQ" in self.model and self.enable_thinking:
            payload.update({
                "enable_thinking": True,
                "thinking_budget": 4096
            })
        
        try:
            response = requests.post(self.chat_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            
            if self.verbose:
                print(f"LLM响应 [会话:{self.session_id[:8]}]: {content}")
            if reasoning:
                print(f"LLM推理过程 [会话:{self.session_id[:8]}]: {reasoning[:200]}...")  # 只打印前200个字符
                
            return {
                "content": content,
                "reasoning": reasoning
            }
            
        except Exception as e:
            if self.verbose:
                print(f"LLM调用失败 [会话:{self.session_id[:8]}]: {e}")
            return {
                "content": "模型调用出现错误，请稍后重试。",
                "reasoning": ""
            }


class SiliconFlowEmbeddings(Embeddings):
    """
    使用SiliconFlow API的嵌入服务
    """

    def __init__(
            self,
            token: str = "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz",
            url: str = "https://api.siliconflow.cn/v1/embeddings",
            model: str = "BAAI/bge-large-zh-v1.5",
            verbose: bool = False
    ):
        super().__init__()
        self.token = token
        self.url = url
        self.model = model
        self.verbose = verbose

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量处理多个文档的向量化"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """处理单个查询文本的向量化"""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> List[float]:
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            if self.verbose:
                print(f"嵌入向量获取失败: {e}")
            # 返回一个默认的向量（全零向量，维度假设为1024）
            return [0.0] * 1024


class StateKnowledgeBase:
    """
    专门管理页面状态信息的知识库
    存储每个测试状态的详细信息：可点击组件、页面结构、测试路径等
    """
    
    def __init__(self, params, verbose=False):
        self.embedding_token = params.get("embedding_token", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.collection_name = params.get("state_collection_name", "state_knowledge")
        self.chunk_size = params.get("state_chunk_size", 800)  # 状态信息较大，使用更大的chunk
        self.chunk_overlap = params.get("state_chunk_overlap", 100)
        self.max_entries = params.get("max_state_entries", 500)
        self.persist_directory = params.get("state_persist_directory", "./state_vectorstore")
        self.verbose = verbose
        self.embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=verbose)
        self.vectorstore = None
        
        # 如果需要清空state知识库
        if params.get("clear_state_on_init", True):
            self.clear_state_vectorstore()
    
    def _initialize_vectorstore(self):
        if self.vectorstore is not None:
            return
            
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print(f"Loaded state KB with {self.vectorstore._collection.count()} documents")
            else:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print("Created new state KB")
        except Exception as e:
            if self.verbose:
                print(f"Failed to initialize state KB: {e}")
            self.vectorstore = Chroma(
                embedding_function=self.embed_model,
                collection_name=self.collection_name
            )

    def clear_state_vectorstore(self):
        """清空状态知识库"""
        if self.verbose:
            print("Clearing state KB...")
        
        # 🚀 使用统一的连接管理
        manage_vectorstore(self.vectorstore, close_connection=True, kb_name="State KB", verbose=self.verbose)
        self.vectorstore = None
        
        import gc
        gc.collect()
        
        # 删除持久化目录
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    print("State KB cleared")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to clear state KB: {e}")
        
        # 重新创建空目录
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except Exception as e:
            if self.verbose:
                print(f"Failed to create state directory: {e}")

    def add_page_state(self, page_title: str, action_list: List, selected_action_index: int, 
                      reasoning: str = "", timestamp: str = None):
        """
        添加详细的页面状态信息到知识库
        """
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            # 详细分析可用动作
            click_actions = []
            input_actions = []
            select_actions = []
            other_actions = []
            
            for i, action in enumerate(action_list):
                action_info = {
                    "index": i,
                    "text": getattr(action, 'text', 'Unknown'),
                    "type": type(action).__name__
                }
                
                if isinstance(action, ClickAction):
                    action_info.update({
                        "action_type": getattr(action, 'action_type', 'unknown'),
                        "addition_info": getattr(action, 'addition_info', '')
                    })
                    click_actions.append(action_info)
                elif isinstance(action, RandomInputAction):
                    action_info.update({
                        "action_type": getattr(action, 'action_type', 'input')
                    })
                    input_actions.append(action_info)
                elif isinstance(action, RandomSelectAction):
                    action_info.update({
                        "action_type": getattr(action, 'action_type', 'select'),
                        "options": getattr(action, 'options', 'N/A')
                    })
                    select_actions.append(action_info)
                else:
                    other_actions.append(action_info)
            
            state_summary = {
                "total_actions": len(action_list),
                "click_actions_count": len(click_actions),
                "input_actions_count": len(input_actions),
                "select_actions_count": len(select_actions),
                "other_actions_count": len(other_actions),
                "selected_action_index": selected_action_index,
                "selected_action_type": type(action_list[selected_action_index]).__name__ if selected_action_index < len(action_list) else "Invalid"
            }
            
            # 构造详细的文档内容
            content = f"""
时间: {timestamp}
页面标题: {page_title}

页面状态概览:
- 总可用动作数: {state_summary['total_actions']}
- 可点击元素: {state_summary['click_actions_count']} 个
- 输入字段: {state_summary['input_actions_count']} 个  
- 选择框: {state_summary['select_actions_count']} 个
- 其他动作: {state_summary['other_actions_count']} 个

选择的动作:
- 索引: {selected_action_index}
- 类型: {state_summary['selected_action_type']}
- 详情: {getattr(action_list[selected_action_index], 'text', 'Unknown') if selected_action_index < len(action_list) else 'Invalid'}

详细可点击元素:
{json.dumps(click_actions, ensure_ascii=False, indent=2)}

详细输入字段:
{json.dumps(input_actions, ensure_ascii=False, indent=2)}

详细选择框:
{json.dumps(select_actions, ensure_ascii=False, indent=2)}

决策推理:
{reasoning[:500]}{'...' if len(reasoning) > 500 else ''}

测试路径分析:
页面 "{page_title}" 提供了 {state_summary['total_actions']} 个交互选项，主要包含 {state_summary['click_actions_count']} 个点击目标和 {state_summary['input_actions_count']} 个输入机会。这种配置适合进行{'导航测试' if state_summary['click_actions_count'] > state_summary['input_actions_count'] else '表单测试'}。
"""
            
            # 创建文档对象
            doc = Document(
                page_content=content,
                metadata={
                    "timestamp": timestamp,
                    "page_title": page_title[:100],
                    "total_actions": state_summary['total_actions'],
                    "click_actions": state_summary['click_actions_count'],
                    "input_actions": state_summary['input_actions_count'],
                    "selected_action_index": selected_action_index,
                    "selected_action_type": state_summary['selected_action_type'],
                    "type": "page_state_record"
                }
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_docs = text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs)
            
            # 持久化
            try:
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
            except Exception:
                pass
            
            if self.verbose:
                print(f"Saved state: {page_title} ({state_summary['total_actions']} actions, {len(split_docs)} chunks)")
            
            # 清理旧记录
            manage_vectorstore(self.vectorstore, self.max_entries, kb_name="State KB", verbose=self.verbose)
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to save page state: {e}")
    
    def _cleanup_old_records(self):
        """使用统一的向量存储管理函数"""
        return manage_vectorstore(self.vectorstore, self.max_entries, kb_name="State KB", verbose=self.verbose)

    def retrieve_similar_states(self, current_page_title: str, action_count: int, k: int = 3) -> str:
        """
        根据页面标题和动作数量检索相似的页面状态信息
        """
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if self.vectorstore is None or self.vectorstore._collection.count() == 0:
                return ""
            
            # 构建查询
            query = f"页面标题 {current_page_title} 动作数量 {action_count} 页面状态 交互选项"
            
            results = self.vectorstore.similarity_search(query, k=k)
            
            if not results:
                return ""
            
            # 组织检索到的状态记录
            similar_states = []
            for doc in results:
                similar_states.append(doc.page_content)
            
            state_context = "\n--- 相似页面状态参考 ---\n" + "\n\n".join(similar_states)
            return state_context
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to retrieve similar states: {e}")
            return ""


class ThinkingKnowledgeBase:
    """
    管理thinking过程的知识库
    """
    
    def __init__(self, params, verbose=False):
        self.embedding_token = params.get("embedding_token", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.collection_name = params.get("thinking_collection_name", "thinking_knowledge")
        self.chunk_size = params.get("thinking_chunk_size", 500)
        self.chunk_overlap = params.get("thinking_chunk_overlap", 50)
        self.max_entries = params.get("max_thinking_entries", 1000)  # 限制知识库大小
        self.persist_directory = params.get("thinking_persist_directory", "./thinking_vectorstore")
        self.verbose = verbose
        self.embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=verbose)
        self.vectorstore = None
        
        # 清理计数器和间隔
        self.cleanup_counter = 0
        self.cleanup_interval = params.get("thinking_cleanup_interval", 20)  # 每20次添加后清理一次
        
        if params.get("clear_thinking_on_init", True):
            self.clear_thinking_vectorstore()
        
    def _initialize_vectorstore(self):
        if self.vectorstore is not None:
            return
            
        try:
            # 尝试加载现有的向量存储
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print(f"Loaded thinking KB with {self.vectorstore._collection.count()} documents")
            else:
                # 创建新的向量存储
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print("Created new thinking KB")
        except Exception as e:
            if self.verbose:
                print(f"Failed to initialize thinking KB: {e}")
            # 创建临时的内存向量存储作为备用
            self.vectorstore = Chroma(
                embedding_function=self.embed_model,
                collection_name=self.collection_name
            )

    def clear_thinking_vectorstore(self):
        """清空Thinking知识库 - 彻底删除所有相关文件"""
        if self.verbose:
            print("Clearing thinking KB...")
        
        # 🚀 使用统一的连接管理
        manage_vectorstore(self.vectorstore, close_connection=True, kb_name="Thinking KB", verbose=self.verbose)
        self.vectorstore = None
        
        # 2. 强制垃圾回收
        import gc
        gc.collect()
        
        # 3. 删除持久化目录
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    print("Thinking KB cleared")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to clear thinking KB: {e}")
        
        # 4. 重新创建空目录
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except Exception as e:
            if self.verbose:
                print(f"Failed to create thinking directory: {e}")
    
    def add_thinking(self, prompt: str, reasoning: str, action_taken: str, timestamp: str = None):
        """
        将thinking过程添加到知识库
        """
        if not reasoning or reasoning.strip() == "":
            return
            
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if timestamp is None:
                timestamp = datetime.now().isoformat()
            
            content = f"""
            时间: {timestamp}
            测试场景: {prompt[:200]}...
            推理过程: {reasoning}
            采取的行动: {action_taken}
            """
            
            # 创建文档对象
            doc = Document(
                page_content=content,
                metadata={
                    "timestamp": timestamp,
                    "prompt_preview": prompt[:100],
                    "action": action_taken,
                    "type": "thinking_record"
                }
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_docs = text_splitter.split_documents([doc])

            self.vectorstore.add_documents(split_docs)
            
            try:
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
            except Exception:
                pass
            
            if self.verbose:
                print(f"Saved thinking: {action_taken[:30]}... ({len(split_docs)} chunks)")
            
            self.cleanup_counter += 1
            if self.cleanup_counter >= self.cleanup_interval:
                manage_vectorstore(self.vectorstore, self.max_entries, kb_name="Thinking KB", verbose=self.verbose)
                self.cleanup_counter = 0
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to save thinking: {e}")
    
    def _cleanup_old_records(self):
        """使用统一的向量存储管理函数"""
        return manage_vectorstore(self.vectorstore, self.max_entries, kb_name="Thinking KB", verbose=self.verbose)

    def retrieve_relevant_thinking(self, query: str, k: int = 3) -> str:
        """
        根据查询检索相关的thinking记录
        """
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if self.vectorstore is None or self.vectorstore._collection.count() == 0:
                return ""
            
            results = self.vectorstore.similarity_search(query, k=k)
            
            if not results:
                return ""
            
            relevant_thinking = []
            for doc in results:
                relevant_thinking.append(doc.page_content)
            
            thinking_context = "\n--- 相关的历史推理经验 ---\n" + "\n\n".join(relevant_thinking)
            return thinking_context
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to retrieve thinking: {e}")
            return ""


class RetrieverInterface:
    def __init__(self, params, verbose=False):
        self.knowledge_path = params.get("knowledge_path", r"C:\Users\ASUS\Desktop\Reasoning+RAG Web Exploration\Make LLM a Testing Expert Bringing Human-like Interaction to.pdf")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.web_testing_pdf = os.path.join(project_root, "agent", "网页测试核心注意事项 (Core Considerations for Web Testing).pdf")
        
        self.chunk_size = params.get("chunk_size", 500)
        self.chunk_overlap = params.get("chunk_overlap", 50)
        self.embedding_token = params.get("embedding_token", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.collection_name = params.get("collection_name", "siliconflow_embed")
        self.top_k = params.get("top_k", 3)
        self.verbose = verbose
        self.persist_directory = params.get("rag_persist_directory", "./rag_vectorstore")
        self._vectorstore = None

    def _close_vectorstore_connections(self):
        """使用统一的连接关闭方法"""
        manage_vectorstore(self._vectorstore, close_connection=True, kb_name="RAG KB", verbose=self.verbose)
        self._vectorstore = None

    def _force_close_sqlite_connections(self):
        """
        强制关闭SQLite连接（改进版）
        """
        try:
            import sqlite3
            import glob
            
            if self.verbose:
                print("Forcing SQLite connections to close...")
            
            # 查找所有相关的SQLite文件
            sqlite_patterns = [
                os.path.join(self.persist_directory, "**/*.sqlite*"),
                os.path.join(self.persist_directory, "**/chroma*"),
            ]
            
            sqlite_files = []
            for pattern in sqlite_patterns:
                sqlite_files.extend(glob.glob(pattern, recursive=True))
            
            if sqlite_files and self.verbose:
                print(f"Found {len(sqlite_files)} database files")
            
            # 等待一段时间让连接自然关闭
            import time
            time.sleep(0.5)
            
        except Exception as e:
            if self.verbose:
                print(f"Error forcing SQLite connections close: {e}")

    def _remove_files_with_retry(self, max_attempts=5, delay=1):
        """
        带重试机制的文件删除（改进版）
        """
        for attempt in range(max_attempts):
            try:
                if not os.path.exists(self.persist_directory):
                    return True
                
                if self.verbose:
                    print(f"Attempting to remove directory (attempt {attempt + 1}/{max_attempts})")
                
                # 删除目录
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    print("RAG vectorstore cleared successfully")
                return True
                
            except Exception as e:
                if attempt < max_attempts - 1:
                    if self.verbose:
                        print(f"Deletion failed, retrying in {delay}s... Error: {e}")
                    import time
                    time.sleep(delay)
                    delay *= 1.5  # 指数退避
                else:
                    if self.verbose:
                        print(f"Failed to delete directory after {max_attempts} attempts: {e}")
                    return False
        
        return False

    def clear_vectorstore(self):
        """
        彻底清空RAG数据库 - 完全改进版
        """
        if self.verbose:
            print("Clearing RAG database...")
        
        # 步骤1: 关闭所有向量存储连接
        self._close_vectorstore_connections()
        
        # 步骤2: 强制垃圾回收
        import gc
        gc.collect()
        
        # 步骤3: 强制关闭SQLite连接
        self._force_close_sqlite_connections()
        
        # 步骤4: 带重试机制的文件删除
        success = self._remove_files_with_retry()
        
        # 步骤5: 重新创建空的持久化目录
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # 验证目录确实为空
            files_remaining = []
            if os.path.exists(self.persist_directory):
                for root, dirs, files in os.walk(self.persist_directory):
                    files_remaining.extend(files)
            
            if not files_remaining:
                if self.verbose:
                    print("RAG database cleared completely")
            else:
                if self.verbose:
                    print(f"Warning: {len(files_remaining)} files still remain")
                    
        except Exception as e:
            if self.verbose:
                print(f"Failed to create persist directory: {e}")

    def _load_vectorstore(self):
        if self._vectorstore is not None:
            return

        if self.verbose:
            print("Initializing RAG knowledge base...")

        possible_paths = [
            self.web_testing_pdf,
            os.path.join(os.getcwd(), "..", "网页测试核心注意事项 (Core Considerations for Web Testing).pdf"),
            os.path.join(os.path.dirname(os.getcwd()), "agent", "网页测试核心注意事项 (Core Considerations for Web Testing).pdf"),
            "../../agent/网页测试核心注意事项 (Core Considerations for Web Testing).pdf"
        ]

        actual_pdf_path = None
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                actual_pdf_path = abs_path
                if self.verbose:
                    print(f"Found PDF: {os.path.basename(abs_path)}")
                break
        
        try:
            all_docs = []
            
            if self.knowledge_path and os.path.exists(self.knowledge_path):
                if self.verbose:
                    print(f"Loading knowledge file: {os.path.basename(self.knowledge_path)}")
                loader = PyPDFLoader(self.knowledge_path)
                pages = loader.load_and_split()
                
                for page in pages:
                    page.metadata["source_file"] = "Knowledge Base"
                
                all_docs.extend(pages)
                if self.verbose:
                    print(f"Loaded {len(pages)} pages")
            
            if actual_pdf_path:
                if self.verbose:
                    print("Loading web testing guidelines")
                loader = PyPDFLoader(actual_pdf_path)
                pages = loader.load_and_split()
                
                for page in pages:
                    page.metadata["source_file"] = "Web Testing Guidelines"
                
                all_docs.extend(pages)
                if self.verbose:
                    print(f"Loaded {len(pages)} pages")
            else:
                if self.verbose:
                    print("Warning: Web testing guidelines PDF not found")
            
            if not all_docs:
                if self.verbose:
                    print("Warning: No knowledge documents found")

                embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=self.verbose)
                self._vectorstore = Chroma(
                    embedding_function=embed_model,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
                return
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            docs = text_splitter.split_documents(all_docs)
            if self.verbose:
                print(f"Document splitting complete: {len(docs)} chunks")
            
            embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=self.verbose)
            self._vectorstore = Chroma.from_documents(
                documents=docs, 
                embedding=embed_model, 
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            
            if self.verbose:
                print("RAG knowledge base initialized successfully")
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to load vectorstore: {e}")

            embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=self.verbose)
            self._vectorstore = Chroma(
                embedding_function=embed_model,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )

    def retrieve(self, prompt: str) -> str:
        try:
            if not self._vectorstore:
                self._load_vectorstore()
            
            if not self._vectorstore:
                if self.verbose:
                    print("Vectorstore not initialized, cannot retrieve")
                return prompt
                
            query = prompt
            result = self._vectorstore.similarity_search(query, k=self.top_k)
            
            if not result:
                if self.verbose:
                    print("No relevant knowledge retrieved")
                return prompt
            
            source_groups = {}
            for doc in result:
                source = doc.metadata.get("source_file", "Unknown")
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append(doc.page_content)
            
            knowledge_sections = []
            for source, contents in source_groups.items():
                section = f"[{source}]:\n" + "\n".join(contents)
                knowledge_sections.append(section)
            
            source_knowledge = "\n\n".join(knowledge_sections)
            
            augmented_prompt = f"""
            使用下面的专业知识来帮助回答查询:
            
            专业知识库:
            {source_knowledge}
            
            查询: 
            {query}
            
            请基于上述专业知识，结合Web测试的最佳实践来回答。
            """
            
            if self.verbose:
                print(f"RAG retrieved {len(result)} chunks from {len(source_groups)} sources")
            
            return augmented_prompt
            
        except Exception as e:
            if self.verbose:
                print(f"RAG retrieval failed: {e}")
            return prompt


class DiversityTracker:
    """
    🎲 多样性跟踪器 - 监控和增强测试的多样性
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.action_diversity_score = 0.0
        self.path_diversity_score = 0.0
        self.exploration_history = []
        self.decision_modes_used = {}  # 跟踪使用的决策模式
        self.exploration_strategies_used = {}  # 跟踪使用的探索策略
        
        # 探索策略权重（动态调整）
        self.exploration_strategy_weights = {
            'conservative': 0.25,    # 保守策略：基于成功经验
            'innovative': 0.35,     # 创新策略：探索新路径
            'balanced': 0.25,       # 平衡策略：综合考虑
            'risk_focused': 0.15    # 风险导向：专注边界测试
        }
        
        # 决策模式权重（动态调整）
        self.decision_mode_weights = {
            'conservative': 0.3,    # 保守决策：基于历史成功
            'exploratory': 0.4,     # 探索决策：尝试新路径
            'balanced': 0.3         # 平衡决策：综合考虑
        }
    
    def select_exploration_strategy(self) -> str:
        """
        🎯 智能选择探索策略，平衡各种策略的使用
        """
        # 计算各策略的使用频率
        total_used = sum(self.exploration_strategies_used.values()) or 1
        strategy_usage = {
            strategy: self.exploration_strategies_used.get(strategy, 0) / total_used
            for strategy in self.exploration_strategy_weights.keys()
        }
        
        # 计算调整后的权重（降低过度使用策略的权重）
        adjusted_weights = {}
        for strategy, base_weight in self.exploration_strategy_weights.items():
            usage_penalty = strategy_usage[strategy] * 0.5  # 使用频率惩罚
            adjusted_weights[strategy] = max(0.1, base_weight - usage_penalty)
        
        # 加权随机选择
        strategies = list(adjusted_weights.keys())
        weights = list(adjusted_weights.values())
        selected_strategy = random.choices(strategies, weights=weights)[0]
        
        # 更新使用计数
        self.exploration_strategies_used[selected_strategy] = self.exploration_strategies_used.get(selected_strategy, 0) + 1
        
        if self.verbose:
            print(f"🎯 选择探索策略: {selected_strategy} (权重: {adjusted_weights[selected_strategy]:.2f})")
        
        return selected_strategy
    
    def select_decision_mode(self) -> str:
        """
        ⚡ 智能选择决策模式，确保决策多样性
        """
        # 计算各模式的使用频率
        total_used = sum(self.decision_modes_used.values()) or 1
        mode_usage = {
            mode: self.decision_modes_used.get(mode, 0) / total_used
            for mode in self.decision_mode_weights.keys()
        }
        
        # 计算调整后的权重
        adjusted_weights = {}
        for mode, base_weight in self.decision_mode_weights.items():
            usage_penalty = mode_usage[mode] * 0.4  # 使用频率惩罚
            adjusted_weights[mode] = max(0.1, base_weight - usage_penalty)
        
        # 加权随机选择
        modes = list(adjusted_weights.keys())
        weights = list(adjusted_weights.values())
        selected_mode = random.choices(modes, weights=weights)[0]
        
        # 更新使用计数
        self.decision_modes_used[selected_mode] = self.decision_modes_used.get(selected_mode, 0) + 1
        
        if self.verbose:
            print(f"⚡ 选择决策模式: {selected_mode} (权重: {adjusted_weights[selected_mode]:.2f})")
        
        return selected_mode
    
    def calculate_action_diversity(self, action_history: List[str]) -> float:
        """
        📊 计算动作多样性分数
        """
        if len(action_history) < 2:
            return 1.0
        
        # 分析最近的动作类型分布
        recent_actions = action_history[-10:]  # 分析最近10个动作
        action_types = {}
        for action in recent_actions:
            action_type = action.split()[0] if action else "Unknown"
            action_types[action_type] = action_types.get(action_type, 0) + 1
        
        # 计算熵作为多样性指标
        total = len(recent_actions)
        entropy = 0
        for count in action_types.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # 归一化到0-1范围
        max_entropy = math.log2(len(action_types)) if action_types else 1
        diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        self.action_diversity_score = diversity_score
        return diversity_score
    
    def get_diversity_feedback(self, action_history: List[str]) -> str:
        """
        🎨 获取多样性反馈建议
        """
        diversity_score = self.calculate_action_diversity(action_history)
        
        if diversity_score < 0.3:
            return """
🎲 **多样性增强建议**：
- 当前测试模式较为单一，建议尝试不同类型的交互
- 考虑探索输入字段、选择框、链接等多种元素
- 尝试不同的测试数据和操作序列
"""
        elif diversity_score < 0.6:
            return """
⚖️ **平衡性建议**：
- 测试多样性适中，可以在当前基础上适度扩展
- 注意平衡探索新功能和验证已知功能
- 适当增加边界条件和异常场景测试
"""
        else:
            return """
🌟 **多样性良好**：
- 当前测试展现了良好的多样性
- 继续保持多元化的测试策略
- 可以深入探索发现的有趣功能点
"""
    
    def get_exploration_enhancement_queries(self, page_title: str, exploration_strategy: str) -> List[str]:
        """
        🔍 根据选择的探索策略生成增强查询
        """
        base_queries = {
            'conservative': [
                f"页面测试分析 {page_title} 功能测试 风险识别",
                f"成功测试案例 {page_title} 最佳实践",
                f"稳定测试策略 验证方法 {page_title}"
            ],
            'innovative': [
                f"创新测试方法 {page_title} 未覆盖测试点",
                f"边界测试 异常场景 {page_title} 探索性测试",
                f"新颖测试路径 发现潜在问题 {page_title}"
            ],
            'balanced': [
                f"综合测试策略 {page_title} 全面覆盖",
                f"平衡探索与验证 {page_title} 测试规划",
                f"多维度测试分析 {page_title} 系统性方法"
            ],
            'risk_focused': [
                f"风险导向测试 {page_title} 安全漏洞",
                f"边界条件测试 极端场景 {page_title}",
                f"错误处理测试 异常情况 {page_title}"
            ]
        }
        
        return base_queries.get(exploration_strategy, base_queries['balanced'])
    
    def get_decision_enhancement_context(self, decision_mode: str) -> str:
        """
        🎯 根据决策模式生成增强上下文
        """
        context_templates = {
            'conservative': """
## 🛡️ 保守决策模式激活
**决策原则**：
- 优先选择历史上成功率高的动作类型
- 基于已验证的测试路径进行决策
- 最小化测试风险，确保稳定推进
- 重点验证核心功能和关键路径

**适用场景**：关键功能验证、稳定性测试、回归测试
""",
            'exploratory': """
## 🚀 探索决策模式激活
**决策原则**：
- 优先尝试历史上较少执行的动作类型
- 勇于探索新的测试路径和功能点
- 适当承担测试风险以获得新发现
- 关注未覆盖的功能区域和边界条件

**适用场景**：功能探索、边界测试、创新路径发现
""",
            'balanced': """
## ⚖️ 平衡决策模式激活
**决策原则**：
- 在稳健测试和探索创新之间找到平衡
- 根据页面重要性动态调整策略
- 综合考虑测试覆盖率和风险控制
- 兼顾验证已知功能和发现新功能

**适用场景**：综合测试、功能验证与探索并重
"""
        }
        
        return context_templates.get(decision_mode, context_templates['balanced'])
    
    def update_diversity_metrics(self, action_taken: str, exploration_strategy: str, decision_mode: str):
        """
        📈 更新多样性指标
        """
        self.exploration_history.append({
            'action': action_taken,
            'exploration_strategy': exploration_strategy,
            'decision_mode': decision_mode,
            'timestamp': datetime.now().isoformat()
        })
        
        # 保持历史记录在合理范围内
        if len(self.exploration_history) > 50:
            self.exploration_history = self.exploration_history[-50:]
    
    def get_diversity_stats(self) -> Dict[str, Any]:
        """
        📊 获取多样性统计信息
        """
        return {
            'action_diversity_score': self.action_diversity_score,
            'exploration_strategies_usage': dict(self.exploration_strategies_used),
            'decision_modes_usage': dict(self.decision_modes_used),
            'total_explorations': len(self.exploration_history),
            'current_weights': {
                'exploration': dict(self.exploration_strategy_weights),
                'decision': dict(self.decision_mode_weights)
            }
        }


class DualModelSystem:
    """
    🚀 双模型协作系统 - R1探索 + QwQ决策
    
    工作流程：
    1. 检测是否到达新状态
    2. 如果是新状态，使用R1进行深度探索分析
    3. 将R1的探索结果存储到专用知识库
    4. 使用QwQ基于探索结果快速做决策
    """
    
    def __init__(self, params, knowledge_bases=None, verbose=False):
        self.verbose = verbose
        
        # 🎲 初始化多样性跟踪器
        self.diversity_tracker = DiversityTracker(verbose=verbose)
        
        # 🚀 RAG知识库系统 - 接收外部传递的知识库实例
        if knowledge_bases:
            self.retriever = knowledge_bases.get('retriever')
            self.state_kb = knowledge_bases.get('state_kb')
            self.thinking_kb = knowledge_bases.get('thinking_kb')
            self.exploration_kb = knowledge_bases.get('exploration_kb')
        else:
            # 如果没有传递知识库，则设为None
            self.retriever = None
            self.state_kb = None
            self.thinking_kb = None
            self.exploration_kb = None
            if verbose:
                print("⚠️ 警告: 未传递知识库实例，RAG增强功能将被禁用")
        
        # R1探索模型配置
        r1_params = params.copy()
        r1_params.update({
            "model": "deepseek-ai/DeepSeek-R1",
            "max_tokens": 2048,  # R1需要更多token用于深度分析
            "temperature": 0.8,  # 稍高温度鼓励探索
            "enable_thinking": True
        })
        
        # QwQ决策模型配置  
        qwq_params = params.copy()
        qwq_params.update({
            "model": "Qwen/QwQ-32B-Preview", 
            "max_tokens": 512,   # QwQ只需少量token做决策
            "temperature": 0.3,  # 低温度确保决策稳定
            "enable_thinking": True
        })
        
        self.r1_explorer = LLMInterface(r1_params, verbose)
        self.qwq_decider = LLMInterface(qwq_params, verbose)
        
        # 状态跟踪
        self.explored_states = set()  # 已探索的状态签名
        self.state_exploration_cache = {}  # 状态探索结果缓存
        self.exploration_count = 0
        
        if verbose:
            print(f"🚀 双模型系统初始化:")
            print(f"   📡 R1探索模型: {r1_params['model']}")
            print(f"   ⚡ QwQ决策模型: {qwq_params['model']}")
            print(f"   🧠 RAG增强: {'启用' if knowledge_bases else '禁用'}")
            print(f"   🎲 多样性跟踪器: 已启用")
    
    def reset_for_new_test(self):
        """重置双模型系统状态"""
        self.r1_explorer.reset_session()
        self.qwq_decider.reset_session()
        self.explored_states.clear()
        self.state_exploration_cache.clear()
        self.exploration_count = 0
        
        if self.verbose:
            print("🔄 双模型系统已重置")
    
    def generate_state_signature(self, page_title: str, action_list: list, html_snippet: str = "") -> str:
        """
        生成页面状态的唯一签名
        """
        # 基于页面标题、动作数量和类型生成签名
        action_types = [type(action).__name__ for action in action_list]
        action_signature = "_".join(sorted(set(action_types)))
        
        # 简化HTML特征(避免过于详细)
        html_features = ""
        if html_snippet:
            # 提取关键HTML标签
            import re
            form_count = len(re.findall(r'<form', html_snippet, re.IGNORECASE))
            input_count = len(re.findall(r'<input', html_snippet, re.IGNORECASE))
            button_count = len(re.findall(r'<button', html_snippet, re.IGNORECASE))
            html_features = f"form{form_count}_input{input_count}_btn{button_count}"
        
        signature = f"{page_title}_{len(action_list)}_{action_signature}_{html_features}"
        return signature[:200]  # 限制长度
    
    def is_new_state(self, state_signature: str) -> bool:
        """检查是否为新状态"""
        return state_signature not in self.explored_states
    
    def r1_explore_state(self, page_title: str, action_list: list, page_context: str, 
                        history_str: str, html: str = "") -> Dict[str, Any]:
        """
        🔍 R1模型深度探索新状态 - 多样性增强版：智能策略选择 + RAG知识增强
        
        返回探索结果字典，包含：
        - analysis: 页面分析
        - strategy: 测试策略
        - recommendations: 推荐动作
        - risk_areas: 风险区域
        - exploration_strategy: 使用的探索策略
        """
        self.exploration_count += 1
        
        # 🎯 智能选择探索策略
        exploration_strategy = self.diversity_tracker.select_exploration_strategy()
        
        # 🚀 多样性增强的RAG知识检索
        if self.verbose:
            print(f"🔍 R1探索 #{self.exploration_count} - 策略: {exploration_strategy}")
            print(f"   开始收集多样化RAG知识...")
        
        # 1. 基于探索策略的多样化专业知识检索
        professional_knowledge = ""
        if self.retriever:
            try:
                # 获取策略相关的多样化查询
                strategy_queries = self.diversity_tracker.get_exploration_enhancement_queries(page_title, exploration_strategy)
                
                # 随机选择2个查询以增加多样性
                selected_queries = random.sample(strategy_queries, k=min(2, len(strategy_queries)))
                
                combined_knowledge = []
                for query in selected_queries:
                    knowledge = self.retriever.retrieve(query)
                    if knowledge:
                        combined_knowledge.append(f"=== {query} ===\n{knowledge}")
                
                professional_knowledge = "\n\n".join(combined_knowledge)
                
                if self.verbose and professional_knowledge:
                    print(f"   📚 获取专业知识: {len(professional_knowledge)} 字符 (策略: {exploration_strategy})")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ 专业知识检索失败: {e}")
        
        # 2. 检索相似页面状态分析经验（增加反向学习）
        similar_states_context = ""
        if self.state_kb:
            try:
                similar_states_context = self.state_kb.retrieve_similar_states(
                    page_title, len(action_list), k=2
                )
                
                # 🎲 20%概率添加失败案例学习（如果有的话）
                if random.random() < 0.2 and self.thinking_kb:
                    failure_query = f"测试失败 错误案例 {page_title} 避免重复"
                    failure_experience = self.thinking_kb.retrieve_relevant_thinking(failure_query, k=1)
                    if failure_experience:
                        similar_states_context += f"\n\n=== 失败案例反向学习 ===\n{failure_experience}"
                
                if self.verbose and similar_states_context:
                    print(f"   📊 获取相似状态: {len(similar_states_context)} 字符")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ 相似状态检索失败: {e}")
        
        # 3. 检索历史分析经验（多角度查询）
        analysis_experience = ""
        if self.thinking_kb:
            try:
                # 基于探索策略的多样化思维查询
                thinking_queries = {
                    'conservative': f"稳健分析 成功经验 {page_title} 测试策略",
                    'innovative': f"创新思路 新颖方法 {page_title} 探索发现",
                    'balanced': f"全面分析 平衡策略 {page_title} 综合考虑", 
                    'risk_focused': f"风险分析 边界测试 {page_title} 安全考量"
                }
                
                thinking_query = thinking_queries.get(exploration_strategy, thinking_queries['balanced'])
                analysis_experience = self.thinking_kb.retrieve_relevant_thinking(thinking_query, k=2)
                
                if self.verbose and analysis_experience:
                    print(f"   🧠 获取分析经验: {len(analysis_experience)} 字符")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ 分析经验检索失败: {e}")
        
        # 🎨 获取多样性反馈
        action_history = [item.get('action', '') for item in self.diversity_tracker.exploration_history]
        diversity_feedback = self.diversity_tracker.get_diversity_feedback(action_history)
        
        # 🔍 RAG增强状态报告
        if self.verbose:
            rag_sources = []
            if professional_knowledge: rag_sources.append("策略知识✓")
            if similar_states_context: rag_sources.append("相似状态✓")
            if analysis_experience: rag_sources.append("分析经验✓")
            
            if rag_sources:
                print(f"   🚀 RAG增强来源: {' '.join(rag_sources)}")
            else:
                print(f"   ⚠️ 未获取到RAG增强数据，使用基础探索模式")
        
        # 构建详细的动作列表
        action_details = []
        for i, action in enumerate(action_list):
            if isinstance(action, ClickAction):
                details = f"{i}. [点击] {getattr(action, 'text', 'Unknown')} (类型: {getattr(action, 'action_type', 'unknown')})"
            elif isinstance(action, RandomInputAction):
                details = f"{i}. [输入] {getattr(action, 'text', 'Unknown')} (字段类型: {getattr(action, 'action_type', 'input')})"
            elif isinstance(action, RandomSelectAction):
                details = f"{i}. [选择] {getattr(action, 'text', 'Unknown')} (选项: {getattr(action, 'options', 'N/A')})"
            else:
                details = f"{i}. [其他] {getattr(action, 'text', 'Unknown')}"
            action_details.append(details)
        
        # 🌟 构建策略导向的探索策略提示
        strategy_guidance = {
            'conservative': """
## 🛡️ 保守探索策略激活
**分析重点**：
- 重点分析核心功能和主要用户流程
- 基于成功经验识别关键测试点
- 评估功能稳定性和可靠性
- 制定渐进式验证策略
""",
            'innovative': """
## 🚀 创新探索策略激活  
**分析重点**：
- 寻找未被充分测试的功能区域
- 探索非常规的用户交互路径
- 识别潜在的创新测试机会
- 设计突破性的测试方法
""",
            'balanced': """
## ⚖️ 平衡探索策略激活
**分析重点**：
- 综合考虑稳定性和创新性
- 平衡深度验证和广度探索
- 兼顾已知功能和未知风险
- 制定全面的测试策略
""",
            'risk_focused': """
## ⚠️ 风险导向探索策略激活
**分析重点**：
- 专注识别潜在的安全风险点
- 分析边界条件和异常场景
- 评估错误处理和容错能力
- 设计风险探测测试方案
"""
        }
        
        current_strategy_guidance = strategy_guidance.get(exploration_strategy, strategy_guidance['balanced'])
        
        # 🎯 构建多样性增强的R1探索prompt
        exploration_prompt = f"""
{professional_knowledge}

{similar_states_context}

{analysis_experience}

{diversity_feedback}

{current_strategy_guidance}

🔍 **R1深度探索任务** - 探索编号 #{self.exploration_count} | 策略: {exploration_strategy.upper()}

你是一位资深的Web测试专家，正在使用 **{exploration_strategy}** 探索策略对当前页面进行深度分析。
请充分利用上述专业知识、相似页面经验、历史分析经验和多样性反馈来指导你的分析。

## 页面信息
{page_context}
{history_str}

## 可用交互元素 ({len(action_list)}个)
{chr(10).join(action_details)}

## 📋 策略导向的深度探索任务
基于 **{exploration_strategy}** 策略，请进行以下分析：

### 1. **策略化页面功能分析**
- 根据{exploration_strategy}策略，深度分析页面的核心功能和技术特点
- 识别与当前策略最匹配的功能区域和测试机会
- 评估页面复杂度和测试优先级（策略导向）

### 2. **多样性测试策略制定**  
- 基于当前多样性状态，制定针对性测试策略
- 参考历史经验和专业知识，确定关键验证点
- 设计多层次测试路径：
  * **{exploration_strategy}导向路径**：符合当前策略的主要测试方向
  * **多样性补充路径**：增强测试覆盖面的辅助方向
  * **创新探索路径**：尝试新的测试角度和方法

### 3. **智能风险识别与机会发现**
- 利用专业知识和{exploration_strategy}策略识别风险点
- 基于相似页面经验预测问题区域
- 发现潜在的测试机会和未覆盖区域

### 4. **策略化动作优先级建议**
请从现有{len(action_list)}个动作中，基于{exploration_strategy}策略确定优先级：
- **策略优先动作** (索引和{exploration_strategy}理由)
- **多样性增强动作** (索引和多样性分析)
- **风险探测动作** (索引和风险评估)
- **创新探索动作** (索引和创新价值)

### 5. **多样化测试数据建议**
基于{exploration_strategy}策略和web测试经验，为输入字段建议：
- **策略导向数据** (符合{exploration_strategy}的测试数据)
- **多样性测试数据** (增加测试覆盖面的数据)
- **边界探索数据** (边界值和异常情况)
- **安全测试数据** (安全漏洞检测数据)

### 6. **策略化探索总结与进化建议**
基于{exploration_strategy}策略和当前分析，总结：
- 本页面在{exploration_strategy}策略下的测试重点和难点
- 与历史相似页面的差异和特殊注意事项
- 策略执行效果评估和调整建议
- 后续探索方向和策略进化建议

**重要提醒**：请充分体现{exploration_strategy}策略的特色，同时考虑多样性增强和测试覆盖的全面性。
提供结构化且具有策略针对性的专业分析结果，这将指导后续的精确测试执行。
"""
        
        if self.verbose:
            print(f"🔍 R1开始{exploration_strategy}策略探索 #{self.exploration_count}: {page_title[:30]}...")
        
        try:
            exploration_result = self.r1_explorer.chat_with_thinking(exploration_prompt)
            
            # 解析探索结果
            analysis_content = exploration_result["content"]
            reasoning_process = exploration_result["reasoning"]
            
            exploration_data = {
                "exploration_id": self.exploration_count,
                "page_title": page_title,
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis_content,
                "reasoning": reasoning_process,
                "action_count": len(action_list),
                "exploration_strategy": exploration_strategy,  # 新增：记录使用的策略
                "diversity_score": self.diversity_tracker.action_diversity_score,  # 新增：多样性分数
                "exploration_prompt": exploration_prompt[:800] + "...",
                "model": "DeepSeek-R1",
                "rag_enhanced": True,
                "strategy_enhanced": True,  # 新增：策略增强标记
                "knowledge_sources": {
                    "professional_knowledge": len(professional_knowledge) > 0,
                    "similar_states": len(similar_states_context) > 0,
                    "analysis_experience": len(analysis_experience) > 0,
                    "diversity_feedback": len(diversity_feedback) > 0
                }
            }
            
            if self.verbose:
                print(f"✅ R1{exploration_strategy}探索完成 #{self.exploration_count}")
                print(f"   📝 分析长度: {len(analysis_content)} 字符")
                print(f"   🧠 推理长度: {len(reasoning_process)} 字符")
                print(f"   🎯 策略: {exploration_strategy}")
                print(f"   🎲 多样性分数: {self.diversity_tracker.action_diversity_score:.2f}")
            
            return exploration_data
            
        except Exception as e:
            if self.verbose:
                print(f"❌ R1{exploration_strategy}探索失败 #{self.exploration_count}: {e}")
            
            # 返回基础探索结果
            return {
                "exploration_id": self.exploration_count,
                "page_title": page_title,
                "timestamp": datetime.now().isoformat(),
                "analysis": f"策略增强探索过程中出现错误: {str(e)}",
                "reasoning": "",
                "action_count": len(action_list),
                "exploration_strategy": exploration_strategy,
                "model": "DeepSeek-R1",
                "rag_enhanced": False,
                "strategy_enhanced": False,
                "error": str(e)
            }
    
    def qwq_decide_action(self, action_list: list, exploration_data: Dict[str, Any], 
                         page_context: str, history_str: str) -> tuple:
        """
        ⚡ QwQ模型基于R1探索结果快速决策 - 多样性增强版：智能决策模式 + RAG经验增强
        
        返回: (action_output, reasoning)
        """
        # 提取R1的关键建议和策略信息
        r1_analysis = exploration_data.get("analysis", "")
        r1_reasoning = exploration_data.get("reasoning", "")
        r1_strategy = exploration_data.get("exploration_strategy", "balanced")
        
        # 🎯 智能选择决策模式
        decision_mode = self.diversity_tracker.select_decision_mode()
        
        # 🚀 多样性增强的RAG知识检索
        if self.verbose:
            print(f"⚡ QwQ决策 - 模式: {decision_mode} | R1策略: {r1_strategy}")
            print(f"   开始收集多样化决策知识...")
        
        # 1. 基于决策模式的历史探索洞察检索
        exploration_insights = ""
        if self.exploration_kb:
            try:
                # 基于决策模式的多样化洞察查询
                insights_queries = {
                    'conservative': f"成功决策案例 稳健选择 {exploration_data.get('page_title', '')} 验证",
                    'exploratory': f"创新决策 探索路径 {exploration_data.get('page_title', '')} 发现",
                    'balanced': f"平衡决策 综合策略 {exploration_data.get('page_title', '')} 执行"
                }
                
                insights_query = insights_queries.get(decision_mode, insights_queries['balanced'])
                exploration_insights = self.exploration_kb.retrieve_exploration_insights(insights_query, k=2)
                
                if self.verbose and exploration_insights:
                    print(f"   🔍 获取探索洞察: {len(exploration_insights)} 字符 (模式: {decision_mode})")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ 探索洞察检索失败: {e}")
        
        
        # 2. 检索相似页面的决策经验（多样性导向）
        similar_decisions = ""
        if self.state_kb:
            try:
                similar_decisions = self.state_kb.retrieve_similar_states(
                    exploration_data.get('page_title', ''), len(action_list), k=2
                )
                
                # 🎲 基于决策模式添加特定类型的经验
                if decision_mode == 'exploratory' and random.random() < 0.3:
                    # 探索模式：30%概率加入失败但有价值的尝试经验
                    if self.thinking_kb:
                        risk_query = f"失败尝试 有价值发现 {exploration_data.get('page_title', '')} 学习"
                        risk_experience = self.thinking_kb.retrieve_relevant_thinking(risk_query, k=1)
                        if risk_experience:
                            similar_decisions += f"\n\n=== 探索失败但有价值的经验 ===\n{risk_experience}"
                
                if self.verbose and similar_decisions:
                    print(f"   📊 获取决策经验: {len(similar_decisions)} 字符")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ 决策经验检索失败: {e}")
        
        # 3. 检索历史执行决策推理（模式导向）
        decision_experience = ""
        if self.thinking_kb:
            try:
                # 基于决策模式的多样化决策查询
                decision_queries = {
                    'conservative': f"稳健决策 成功执行 {exploration_data.get('page_title', '')} 验证",
                    'exploratory': f"探索决策 创新尝试 {exploration_data.get('page_title', '')} 发现",
                    'balanced': f"平衡决策 综合考虑 {exploration_data.get('page_title', '')} 执行"
                }
                
                decision_query = decision_queries.get(decision_mode, decision_queries['balanced'])
                decision_experience = self.thinking_kb.retrieve_relevant_thinking(decision_query, k=2)
                
                if self.verbose and decision_experience:
                    print(f"   🧠 获取决策推理: {len(decision_experience)} 字符")
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ 决策推理检索失败: {e}")
        
        # 🎨 获取决策增强上下文
        decision_enhancement = self.diversity_tracker.get_decision_enhancement_context(decision_mode)
        
        # 📊 计算动作历史多样性
        action_history = [item.get('action', '') for item in self.diversity_tracker.exploration_history]
        diversity_feedback = self.diversity_tracker.get_diversity_feedback(action_history)
        
        # 🔍 RAG增强状态报告
        if self.verbose:
            rag_sources = []
            if exploration_insights: rag_sources.append("探索洞察✓")
            if similar_decisions: rag_sources.append("决策经验✓")
            if decision_experience: rag_sources.append("推理经验✓")
            
            if rag_sources:
                print(f"   🚀 RAG增强来源: {' '.join(rag_sources)}")
            else:
                print(f"   ⚠️ 未获取到RAG增强数据，使用基础决策模式")
        
        # 构建动作列表（增加多样性分析）
        action_analysis = []
        action_types_count = {}
        
        for i, action in enumerate(action_list):
            action_simple = self.format_action_simple(action)
            action_type = action_simple.split()[0] if action_simple else "Unknown"
            action_types_count[action_type] = action_types_count.get(action_type, 0) + 1
            
            # 计算该动作类型在历史中的使用频率
            historical_usage = sum(1 for hist_action in action_history if hist_action.startswith(action_type))
            usage_indicator = "🔥" if historical_usage > 3 else "🆕" if historical_usage == 0 else "📊"
            
            action_analysis.append(f"{i}. {action_simple} {usage_indicator}")
        
        action_list_str = "\n".join(action_analysis)
        
        # 🌟 构建决策模式导向的多样性分析
        diversity_analysis = f"""
## 🎲 当前多样性状态分析
- **多样性分数**: {self.diversity_tracker.action_diversity_score:.2f}/1.0
- **可用动作类型**: {dict(action_types_count)}
- **历史使用频率**: 🔥频繁 📊适中 🆕未使用
- **决策模式**: {decision_mode.upper()}
- **R1探索策略**: {r1_strategy.upper()}

{diversity_feedback}
"""
        
        # 🎯 构建多样性增强的QwQ决策prompt
        decision_prompt = f"""
{exploration_insights}

{similar_decisions}

{decision_experience}

{decision_enhancement}

{diversity_analysis}

⚡ **QwQ智能决策任务** - 决策模式: {decision_mode.upper()}

基于R1模型的{r1_strategy}策略探索分析和历史决策经验，使用{decision_mode}决策模式做出最优选择。
请充分利用上述探索洞察、相似决策经验、历史推理和多样性分析来指导你的决策。

## 当前状态
{page_context}
{history_str}

## R1深度探索分析（策略: {r1_strategy}）
{r1_analysis[:1000]}...

## 可选动作分析 ({len(action_list)}个)
{action_list_str}

## 🎯 {decision_mode.upper()}模式智能决策
基于{decision_mode}决策模式、R1的{r1_strategy}策略分析和多样性状态，选择当前最合适的动作：

### 决策优先级框架
1. **模式导向优先级** - 符合{decision_mode}模式的决策原则
2. **R1策略契合度** - 与R1的{r1_strategy}策略分析的一致性
3. **多样性增强价值** - 对提升测试多样性的贡献
4. **历史经验指导** - 相似场景下的成功经验参考
5. **风险收益平衡** - 测试价值与执行风险的权衡

### 多样性决策考虑因素
- **动作类型多样性**: 选择历史使用较少的动作类型 (🆕 > 📊 > 🔥)
- **探索路径创新**: 尝试与历史路径不同的测试方向
- **功能覆盖均衡**: 平衡不同功能区域的测试覆盖
- **策略模式协调**: {decision_mode}模式与{r1_strategy}策略的最佳结合

### {decision_mode.upper()}模式特定指导
根据{decision_mode}决策模式：
- **Conservative模式**: 优先选择历史成功率高、风险可控的动作
- **Exploratory模式**: 勇于尝试新路径、未使用的动作类型
- **Balanced模式**: 在稳健验证和创新探索之间找到最佳平衡点

### 输出要求
**严格按照以下格式输出**：
- 点击动作：直接返回数字，如 "3"
- 输入动作：返回"数字:文本"，如 "5:test@example.com"

### 决策说明（简短）
在一行内简要说明选择理由，格式：
"{decision_mode}模式-{action_type}-理由"

**决策原则**：
- 基于{decision_mode}模式的特定优势和R1的{r1_strategy}策略洞察
- 优化测试多样性和覆盖面
- 确保决策的创新性和执行效果

请基于上述全面分析快速做出精准决策，只返回动作索引或"索引:文本"格式。
"""
        
        if self.verbose:
            print(f"⚡ QwQ开始{decision_mode}模式智能决策...")
        
        try:
            decision_result = self.qwq_decider.chat_with_thinking(decision_prompt)
            qwq_output = decision_result["content"]
            qwq_reasoning = decision_result["reasoning"]
            
            # 🎲 更新多样性追踪
            action_taken = f"{decision_mode}决策模式选择"
            self.diversity_tracker.update_diversity_metrics(
                action_taken, r1_strategy, decision_mode
            )
            
            if self.verbose:
                print(f"⚡ QwQ{decision_mode}决策输出: {qwq_output}")
                print(f"🧠 QwQ推理: {qwq_reasoning[:100]}...")
                print(f"🎯 模式: {decision_mode} | R1策略: {r1_strategy}")
                print(f"🎲 多样性分数: {self.diversity_tracker.action_diversity_score:.2f}")
            
            return qwq_output, qwq_reasoning
            
        except Exception as e:
            if self.verbose:
                print(f"❌ QwQ{decision_mode}决策失败: {e}")
            return None, f"多样性增强决策错误: {str(e)}"
    
    def format_action_simple(self, action) -> str:
        """简化的动作格式化"""
        if isinstance(action, ClickAction):
            return f"点击 '{getattr(action, 'text', 'Unknown')}'"
        elif isinstance(action, RandomInputAction):
            return f"输入 '{getattr(action, 'text', 'Unknown')}'"
        elif isinstance(action, RandomSelectAction):
            return f"选择 '{getattr(action, 'text', 'Unknown')}'"
        else:
            return f"操作 '{getattr(action, 'text', 'Unknown')}'"
    
    def get_dual_model_stats(self) -> dict:
        """获取双模型系统统计（包含多样性信息）"""
        diversity_stats = self.diversity_tracker.get_diversity_stats()
        
        return {
            "total_explorations": self.exploration_count,
            "cached_states": len(self.explored_states),
            "cache_size": len(self.state_exploration_cache),
            "r1_session_id": self.r1_explorer.session_id[:8],
            "qwq_session_id": self.qwq_decider.session_id[:8],
            # 🎲 多样性统计
            "diversity_metrics": diversity_stats,
            "current_diversity_score": diversity_stats.get('action_diversity_score', 0.0),
            "exploration_strategies_usage": diversity_stats.get('exploration_strategies_usage', {}),
            "decision_modes_usage": diversity_stats.get('decision_modes_usage', {})
        }


class ExplorationKnowledgeBase:
    """
    🔍 专门存储R1探索结果的知识库
    """
    
    def __init__(self, params, verbose=False):
        self.embedding_token = params.get("embedding_token", "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz")
        self.collection_name = params.get("exploration_collection_name", "exploration_knowledge")
        self.chunk_size = params.get("exploration_chunk_size", 1000)  # 探索结果较大
        self.chunk_overlap = params.get("exploration_chunk_overlap", 150)
        self.max_entries = params.get("max_exploration_entries", 200)
        self.persist_directory = params.get("exploration_persist_directory", "./exploration_vectorstore")
        self.verbose = verbose
        self.embed_model = SiliconFlowEmbeddings(token=self.embedding_token, verbose=verbose)
        self.vectorstore = None
        
        if params.get("clear_exploration_on_init", True):
            self.clear_exploration_vectorstore()
    
    def _initialize_vectorstore(self):
        if self.vectorstore is not None:
            return
            
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print(f"Loaded exploration KB with {self.vectorstore._collection.count()} documents")
            else:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embed_model,
                    collection_name=self.collection_name
                )
                if self.verbose:
                    print("Created new exploration KB")
        except Exception as e:
            if self.verbose:
                print(f"Failed to initialize exploration KB: {e}")
            self.vectorstore = Chroma(
                embedding_function=self.embed_model,
                collection_name=self.collection_name
            )
    
    def clear_exploration_vectorstore(self):
        """清空探索知识库"""
        if self.verbose:
            print("Clearing exploration KB...")
        
        manage_vectorstore(self.vectorstore, close_connection=True, kb_name="Exploration KB", verbose=self.verbose)
        self.vectorstore = None
        
        import gc
        gc.collect()
        
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                if self.verbose:
                    print("Exploration KB cleared")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to clear exploration KB: {e}")
        
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
        except Exception as e:
            if self.verbose:
                print(f"Failed to create exploration directory: {e}")
    
    def add_exploration_result(self, exploration_data: Dict[str, Any]):
        """添加R1探索结果到知识库"""
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
            
            # 构建探索文档内容
            content = f"""
探索ID: {exploration_data.get('exploration_id', 'Unknown')}
时间: {exploration_data.get('timestamp', 'Unknown')}
页面: {exploration_data.get('page_title', 'Unknown')}
动作数量: {exploration_data.get('action_count', 0)}
使用模型: {exploration_data.get('model', 'Unknown')}

=== R1深度分析 ===
{exploration_data.get('analysis', '')}

=== R1推理过程 ===
{exploration_data.get('reasoning', '')}

=== 探索摘要 ===
这是一次针对"{exploration_data.get('page_title', 'Unknown')}"页面的深度探索，
发现了{exploration_data.get('action_count', 0)}个可交互元素，
由DeepSeek-R1模型进行专业分析和测试策略制定。
"""
            
            doc = Document(
                page_content=content,
                metadata={
                    "exploration_id": exploration_data.get('exploration_id'),
                    "timestamp": exploration_data.get('timestamp'),
                    "page_title": exploration_data.get('page_title', '')[:100],
                    "action_count": exploration_data.get('action_count', 0),
                    "model": exploration_data.get('model', ''),
                    "type": "r1_exploration_result"
                }
            )
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            split_docs = text_splitter.split_documents([doc])
            self.vectorstore.add_documents(split_docs)
            
            try:
                if hasattr(self.vectorstore, 'persist'):
                    self.vectorstore.persist()
            except Exception:
                pass
            
            if self.verbose:
                print(f"💾 保存R1探索结果 #{exploration_data.get('exploration_id')} ({len(split_docs)} chunks)")
            
            # 清理旧记录
            manage_vectorstore(self.vectorstore, self.max_entries, kb_name="Exploration KB", verbose=self.verbose)
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to save exploration result: {e}")
    
    def retrieve_exploration_insights(self, query: str, k: int = 2) -> str:
        """检索R1探索洞察"""
        try:
            if self.vectorstore is None:
                self._initialize_vectorstore()
                
            if self.vectorstore is None or self.vectorstore._collection.count() == 0:
                return ""
            
            results = self.vectorstore.similarity_search(query, k=k)
            
            if not results:
                return ""
            
            insights = []
            for doc in results:
                insights.append(doc.page_content)
            
            exploration_context = "\n--- R1探索洞察 ---\n" + "\n\n".join(insights)
            return exploration_context
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to retrieve exploration insights: {e}")
            return ""


class rag_llm_agent(Agent):
    def __init__(self, params):
        self.params = params
        self.verbose = params.get("verbose", False)
        
        # 🚀 先初始化所有知识库系统
        self.retriever = RetrieverInterface(params, verbose=self.verbose)
        self.thinking_kb = ThinkingKnowledgeBase(params, verbose=self.verbose)
        self.state_kb = StateKnowledgeBase(params, verbose=self.verbose)
        self.exploration_kb = ExplorationKnowledgeBase(params, verbose=self.verbose)
        
        # 🧠 将所有知识库打包传递给双模型系统
        knowledge_bases = {
            'retriever': self.retriever,
            'state_kb': self.state_kb,
            'thinking_kb': self.thinking_kb,
            'exploration_kb': self.exploration_kb
        }
        
        # 🚀 初始化双模型协作系统 - 传递知识库实例
        self.dual_model_system = DualModelSystem(params, knowledge_bases=knowledge_bases, verbose=self.verbose)
        
        self.app_name = params.get("app_name", "Web Testing")
        self.history = []
        self.max_history_length = params.get("max_history_length", 5)

        self.login_state = "none"  # none, detected, username_filled, password_filled, completed
        self.login_credentials = {
            "username": "Nefelibata-Zhu",
            "password": "han19780518"
        }
        self.login_attempts = 0
        self.max_login_attempts = 3

        self.explored_pages = set()
        self.executed_actions = {}
        self.page_action_history = {}
        self.bug_indicators = []

        if params.get("reset_dual_model_on_init", True):
            if self.verbose:
                print("Resetting dual model system on init...")
            self.reset_dual_model_session()

        if params.get("clear_rag_on_init", True):
            if self.verbose:
                print("Clearing all RAG databases on init...")
            self.clear_all_rag_databases()

        if self.verbose:
            print(f"🚀 双模型RAG Agent initialized for {self.app_name}")
            stats = self.dual_model_system.get_dual_model_stats()
            print(f"   📡 R1会话: {stats['r1_session_id']}")
            print(f"   ⚡ QwQ会话: {stats['qwq_session_id']}")
            print("🧠 R1探索 + QwQ决策 协作系统已启用")
            print("🚀 四层RAG知识库系统已启用:")
            print("   📖 RetrieverKB: 专业测试知识文档")
            print("   🧠 ThinkingKB: 模型推理过程记录")
            print("   📊 StateKB: 页面状态和交互历史")
            print("   🔍 ExplorationKB: R1探索结果专用存储")

    def reset_login_state(self):
        self.login_state = "none"
        self.login_attempts = 0
        if self.verbose:
            print("登录状态已重置")

    def set_login_credentials(self, username: str, password: str):
        """
        设置登录凭证
        
        Args:
            username: 用户名或邮箱
            password: 密码
        """
        self.login_credentials["username"] = username
        self.login_credentials["password"] = password
        if self.verbose:
            print(f"🔐 登录凭证已更新 - 用户名: {username}")

    def handle_smart_login(self, action_list, page_title: str = "") -> WebAction:
        """
        🔐 智能登录状态机 - 自动完成登录流程
        """
        if self.verbose:
            print(f"🔐 智能登录处理 - 当前状态: {self.login_state}")

        # 分析可用动作
        username_actions = []
        password_actions = []
        login_button_actions = []
        
        for i, action in enumerate(action_list):
            if isinstance(action, RandomInputAction):
                field_text = action.text.lower()
                if any(keyword in field_text for keyword in ["username", "email", "user"]):
                    username_actions.append((i, action))
                elif any(keyword in field_text for keyword in ["password", "pwd"]):
                    password_actions.append((i, action))
            elif isinstance(action, ClickAction):
                click_text = action.text.lower()
                if any(keyword in click_text for keyword in ["sign in", "login", "log in", "登录"]):
                    login_button_actions.append((i, action))

        # 状态机逻辑
        if self.login_state == "none" or self.login_state == "detected":
            # 第一步：填写用户名
            if username_actions:
                self.login_state = "username_filling"
                action_index, selected_action = username_actions[0]
                
                if hasattr(selected_action, 'set_input_text'):
                    selected_action.set_input_text(self.login_credentials["username"])
                
                action_description = f"🔐 自动填入用户名: {self.login_credentials['username']}"
                self.history.append(action_description)
                
                if self.verbose:
                    print(f"🔐 步骤1: 填入用户名 - Action [{action_index}]")
                print(f"Action [{action_index}]: 🔐 AUTO LOGIN - Username input")
                
                self.login_state = "username_filled"
                return selected_action

        elif self.login_state == "username_filled":
            # 第二步：填写密码
            if password_actions:
                self.login_state = "password_filling"
                action_index, selected_action = password_actions[0]
                
                if hasattr(selected_action, 'set_input_text'):
                    selected_action.set_input_text(self.login_credentials["password"])
                
                action_description = f"🔐 自动填入密码: {'*' * len(self.login_credentials['password'])}"
                self.history.append(action_description)
                
                if self.verbose:
                    print(f"🔐 步骤2: 填入密码 - Action [{action_index}]")
                print(f"Action [{action_index}]: 🔐 AUTO LOGIN - Password input")
                
                self.login_state = "password_filled"
                return selected_action

        elif self.login_state == "password_filled":
            # 第三步：点击登录按钮
            if login_button_actions:
                action_index, selected_action = login_button_actions[0]
                
                action_description = "🔐 自动点击登录按钮"
                self.history.append(action_description)
                
                if self.verbose:
                    print(f"🔐 步骤3: 点击登录按钮 - Action [{action_index}]")
                print(f"Action [{action_index}]: 🔐 AUTO LOGIN - Click login button")
                
                self.login_state = "completed"
                return selected_action

        # 如果没有找到对应的动作，增加尝试次数
        self.login_attempts += 1
        if self.login_attempts >= self.max_login_attempts:
            if self.verbose:
                print(f"🔐 登录尝试失败 {self.max_login_attempts} 次，切换到普通模式")
            self.login_state = "completed"  # 强制完成，使用普通逻辑
            return None
        
        # 返回 None 表示继续尝试
        return None

    def clear_rag_database(self):
        """
        清空三个RAG数据库
        """
        try:
            if self.verbose:
                print("Clearing all RAG databases...")
            self.retriever.clear_vectorstore()
            self.thinking_kb.clear_thinking_vectorstore()
            self.state_kb.clear_state_vectorstore()
            
            if self.verbose:
                print("All RAG databases cleared successfully")
        except Exception as e:
            if self.verbose:
                print(f"Failed to clear RAG databases: {e}")

    def reset_llm_session(self):
        """
        重置LLM会话状态 - 确保测试独立性
        """
        try:
            self.dual_model_system.r1_explorer.reset_session()
            self.dual_model_system.qwq_decider.reset_session()
            if self.verbose:
                print("LLM session reset successfully")
        except Exception as e:
            if self.verbose:
                print(f"Failed to reset LLM session: {e}")

    def reset_for_new_test(self):
        """
        为新测试重置Agent状态 - 确保完全独立的测试环境
        """
        if self.verbose:
            print("Resetting agent for new test...")
        
        # 1. 清空历史记录
        self.history = []
        
        # 2. 重置双模型系统状态
        self.reset_dual_model_session()
        
        # 3. 清空所有RAG数据库（包括状态知识库）
        self.clear_rag_database()
        
        if self.verbose:
            print(f"Agent reset complete - New session: {self.dual_model_system.r1_explorer.session_id[:8]}")

    def format_action_info(self, action):
        """格式化动作信息"""
        if isinstance(action, ClickAction):
            operation_name = "Click"
            # 安全地获取属性
            action_type = getattr(action, 'action_type', 'unknown')
            addition_info = getattr(action, 'addition_info', '')
            details = f"Widget text: '{action.text}', type: {action_type}, info: {addition_info}"
        elif isinstance(action, RandomInputAction):
            operation_name = "Input"
            # 安全地获取属性
            action_type = getattr(action, 'action_type', 'input')
            details = f"Input field: '{action.text}', type: {action_type}"
        elif isinstance(action, RandomSelectAction):
            operation_name = "Select"
            # 安全地获取属性
            action_type = getattr(action, 'action_type', 'select')
            options = getattr(action, 'options', 'N/A')
            details = f"Select field: '{action.text}', type: {action_type}, options: {options}"
        else:
            operation_name = "UnknownOperation"
            # 安全地获取text属性
            text = getattr(action, 'text', 'Unknown')
            details = f"Widget: '{text}'"

        return f"'{operation_name}' on {details}"

    def detect_login_page(self, action_list, html: str = "", page_title: str = "") -> bool:
        """
        检测是否为登录页面
        """
        # 检查页面标题
        login_title_keywords = [
            "sign in", "login", "log in", "登录", "登入"
        ]
        
        if page_title:
            title_lower = page_title.lower()
            for keyword in login_title_keywords:
                if keyword in title_lower:
                    return True
        
        # 检查HTML内容
        if html:
            html_lower = html.lower()
            if "sign in to github" in html_lower or "github login" in html_lower:
                return True
        
        # 检查动作列表中是否有登录相关的输入字段
        login_field_keywords = [
            "username", "email", "password", "login", "sign in"
        ]
        
        input_fields = [action for action in action_list if isinstance(action, RandomInputAction)]
        login_field_count = 0
        
        for action in input_fields:
            field_text = action.text.lower()
            for keyword in login_field_keywords:
                if keyword in field_text:
                    login_field_count += 1
                    break
        
        # 如果有2个或以上的登录相关字段，认为是登录页面
        return login_field_count >= 2

    def get_action_signature(self, action) -> str:
        """
        生成动作的唯一签名，用于去重
        """
        if hasattr(action, 'text') and hasattr(action, 'location'):
            return f"{type(action).__name__}_{action.text}_{action.location}"
        elif hasattr(action, 'text'):
            return f"{type(action).__name__}_{action.text}"
        else:
            return f"{type(action).__name__}_{str(action)}"

    def update_exploration_history(self, current_url: str, selected_action, action_index: int):
        """
        更新探索历史记录
        """
        # 记录访问的页面
        self.explored_pages.add(current_url)
        
        # 记录执行的动作
        if current_url not in self.executed_actions:
            self.executed_actions[current_url] = {}
        
        action_signature = self.get_action_signature(selected_action)
        if action_signature not in self.executed_actions[current_url]:
            self.executed_actions[current_url][action_signature] = 0
        self.executed_actions[current_url][action_signature] += 1
        
        # 记录页面动作历史
        if current_url not in self.page_action_history:
            self.page_action_history[current_url] = []
        
        self.page_action_history[current_url].append({
            "action_text": getattr(selected_action, 'text', 'Unknown'),
            "action_type": type(selected_action).__name__,
            "action_index": action_index,
            "timestamp": datetime.now().isoformat(),
            "execution_count": self.executed_actions[current_url][action_signature]
        })
        
        # 保持历史记录在合理范围内
        if len(self.page_action_history[current_url]) > 20:
            self.page_action_history[current_url] = self.page_action_history[current_url][-20:]

    def generate_login_focused_prompt(self, action_list, page_context: str, history_str: str, page_title: str = "") -> str:
        """
        生成专注于登录的提示词
        """
        descriptions = [f"{i}. " + self.format_action_info(a) for i, a in enumerate(action_list)]
        action_descriptions_str = "\n".join(descriptions)
        
        # 识别登录相关的动作
        username_actions = []
        password_actions = []
        login_button_actions = []
        
        for i, action in enumerate(action_list):
            if isinstance(action, RandomInputAction):
                field_text = action.text.lower()
                if any(keyword in field_text for keyword in ["username", "email", "user"]):
                    username_actions.append(i)
                elif any(keyword in field_text for keyword in ["password", "pwd"]):
                    password_actions.append(i)
            elif isinstance(action, ClickAction):
                click_text = action.text.lower()
                if any(keyword in click_text for keyword in ["sign in", "login", "log in", "登录"]):
                    login_button_actions.append(i)
        
        login_suggestions = ""
        if username_actions:
            login_suggestions += f"\n用户名输入字段索引: {username_actions} (输入: Nefelibata-Zhu)"
        if password_actions:
            login_suggestions += f"\n密码输入字段索引: {password_actions} (输入: han19780518)"
        if login_button_actions:
            login_suggestions += f"\n登录按钮索引: {login_button_actions}"
        
        return f"""
检测到登录页面！请立即完成登录流程。
        
        {page_context}
        {history_str}
        
        可以操作的界面元素有:
        {action_descriptions_str}
        
登录操作建议：{login_suggestions}
        
登录步骤（请严格按顺序执行）：
        1. 找到用户名/邮箱输入字段 → 输入"Nefelibata-Zhu"
        2. 找到密码输入字段 → 输入"han19780518"
        3. 点击"Sign in"登录按钮（避免注册按钮）
        
重要提醒：
        - 优先完成登录，不要进行其他操作
        - 确保使用正确的凭据格式
        - 避免点击注册相关按钮
        
        请返回一个数字，对应上面列表中动作的索引。
        如果选择输入动作，格式为"索引:文本"。
        例如：用户名字段输入，返回"{username_actions[0] if username_actions else 'X'}:Nefelibata-Zhu"
        
        只返回索引数字或"索引:文本"格式，不要返回任何其他解释。
        """.strip()

    def get_action(self, web_state: WebState, html: str) -> WebAction:
        """
        🚀 双模型协作决策方法 - R1探索 + QwQ决策
        """
        action_list = web_state.get_action_list()
        if self.verbose:
            print(f"Available actions: {len(action_list)}")
        if not action_list:
            if self.verbose:
                print("Warning: No available actions")
            return None

        # 获取页面上下文信息
        page_context = ""
        page_title = ""
        current_url = ""
        if html:
            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
            if title_match:
                page_title = title_match.group(1)
                page_context = f"当前页面标题: {page_title}\n"
        
        if hasattr(web_state, 'url'):
            current_url = web_state.url
        elif page_title:
            current_url = page_title

        # 构建历史记录字符串
        history_str = ""
        if self.history:
            history_str = "最近的操作历史:\n" + "\n".join([f"- {h}" for h in self.history[-self.max_history_length:]])

        # 检测是否为登录页面
        is_login_page = self.detect_login_page(action_list, html, page_title)
        
        # 🔐 智能登录处理 - 优先使用状态机
        if is_login_page and self.login_state != "completed":
            if self.login_state == "none":
                self.login_state = "detected"
                print("🔐 检测到登录页面，启动智能登录状态机")
            
            smart_login_action = self.handle_smart_login(action_list, page_title)
            if smart_login_action is not None:
                return smart_login_action
            elif self.login_state == "completed":
                print("🔐 登录流程已完成，切换到双模型测试模式")
                self.reset_login_state()
        
        if not is_login_page and self.login_state != "none":
            if self.verbose:
                print("🔐 离开登录页面，重置登录状态")
            self.reset_login_state()

        # 🚀 双模型协作决策
        print("🚀 使用双模型协作系统进行决策")
        
        try:
            # 1. 生成状态签名
            state_signature = self.dual_model_system.generate_state_signature(
                page_title, action_list, html[:500]  # 只使用前500字符避免过长
            )
            
            # 2. 检查是否为新状态
            is_new_state = self.dual_model_system.is_new_state(state_signature)
            
            if is_new_state:
                # 3a. 新状态：使用R1进行深度探索
                print(f"🔍 检测到新状态，启动R1深度探索")
                
                exploration_data = self.dual_model_system.r1_explore_state(
                    page_title=page_title or "Unknown Page",
                    action_list=action_list,
                    page_context=page_context,
                    history_str=history_str,
                    html=html
                )
                
                # 保存探索结果到专用知识库
                self.exploration_kb.add_exploration_result(exploration_data)
                
                # 标记状态为已探索
                self.dual_model_system.explored_states.add(state_signature)
                self.dual_model_system.state_exploration_cache[state_signature] = exploration_data
                
                # 3b. 基于R1探索结果，使用QwQ快速决策
                qwq_output, qwq_reasoning = self.dual_model_system.qwq_decide_action(
                    action_list=action_list,
                    exploration_data=exploration_data,
                    page_context=page_context,
                    history_str=history_str
                )
                
                combined_reasoning = f"R1探索: {exploration_data.get('reasoning', '')[:200]}... | QwQ决策: {qwq_reasoning[:200]}..."
                
            else:
                # 4. 已知状态：直接使用QwQ基于缓存的探索结果决策
                print(f"⚡ 已知状态，使用QwQ快速决策")
                
                cached_exploration = self.dual_model_system.state_exploration_cache.get(state_signature)
                if cached_exploration:
                    qwq_output, qwq_reasoning = self.dual_model_system.qwq_decide_action(
                        action_list=action_list,
                        exploration_data=cached_exploration,
                        page_context=page_context,
                        history_str=history_str
                    )
                else:
                    # 缓存丢失，快速生成基础探索信息
                    basic_exploration = {
                        "analysis": f"基础状态分析：页面有{len(action_list)}个可交互元素",
                        "reasoning": "使用基础探索信息进行快速决策"
                    }
                    qwq_output, qwq_reasoning = self.dual_model_system.qwq_decide_action(
                        action_list=action_list,
                        exploration_data=basic_exploration,
                        page_context=page_context,
                        history_str=history_str
                    )
                
                combined_reasoning = f"缓存决策: {qwq_reasoning[:300]}..."

            if self.verbose:
                print(f"双模型输出: {qwq_output}")

            # 5. 解析QwQ输出
            action_index, input_text = self.parse_output(qwq_output, len(action_list))

            if action_index is not None and 0 <= action_index < len(action_list):
                # 决策有效
                selected_action = action_list[action_index]
                
                if isinstance(selected_action, RandomInputAction) and input_text:
                    if hasattr(selected_action, 'set_input_text'):
                        selected_action.set_input_text(input_text)

                # 更新探索历史记录
                self.update_exploration_history(current_url, selected_action, action_index)
                
                action_description = self.format_action_info(selected_action)
                if input_text:
                    action_description += f" with input: '{input_text}'"
                
                self.history.append(action_description)

                # 保存知识库信息
                self.state_kb.add_page_state(
                    page_title=page_title or "Unknown Page",
                    action_list=action_list,
                    selected_action_index=action_index,
                    reasoning=combined_reasoning
                )

                self.thinking_kb.add_thinking(
                    prompt=f"双模型协作: 新状态={is_new_state}",
                    reasoning=combined_reasoning,
                    action_taken=action_description
                )

                if len(self.history) > self.max_history_length:
                    self.history = self.history[-self.max_history_length:]

                # 显示决策结果
                model_info = "🔍R1+⚡QwQ" if is_new_state else "⚡QwQ"
                if self.verbose:
                    print(f"✅ {model_info}选择 [{action_index}]: {action_description}")
                    # 🎲 显示多样性统计
                    diversity_stats = self.dual_model_system.get_dual_model_stats()
                    print(f"🎲 多样性分数: {diversity_stats['current_diversity_score']:.2f}")
                    print(f"🎯 探索策略使用: {diversity_stats['exploration_strategies_usage']}")
                    print(f"⚡ 决策模式使用: {diversity_stats['decision_modes_usage']}")
                else:
                    print(f"Action [{action_index}]: {self.format_action_info(selected_action)} ({model_info})")
                
                return selected_action
            else:
                # 决策无效，随机回退
                if self.verbose:
                    print(f"⚠️ 双模型选择 {action_index} 无效，使用随机回退策略")
                
                import random
                fallback_index = random.randint(0, len(action_list) - 1)
                fallback_action = action_list[fallback_index]
                
                self.update_exploration_history(current_url, fallback_action, fallback_index)
                
                fallback_description = f"Random Fallback: {self.format_action_info(fallback_action)} 🎲"
                
                self.state_kb.add_page_state(
                    page_title=page_title or "Unknown Page",
                    action_list=action_list,
                    selected_action_index=fallback_index,
                    reasoning=f"双模型选择无效，随机回退: {combined_reasoning if 'combined_reasoning' in locals() else 'Random selection'}"
                )

                if 'combined_reasoning' in locals():
                    self.thinking_kb.add_thinking(
                        prompt="双模型决策失败",
                        reasoning=combined_reasoning,
                        action_taken=fallback_description
                    )

                self.history.append(fallback_description)
                if len(self.history) > self.max_history_length:
                    self.history = self.history[-self.max_history_length:]
                
                return fallback_action

        except Exception as e:
            if self.verbose:
                print(f"❌ 双模型协作失败: {e}，使用随机回退")
            
            # 随机回退
            import random
            fallback_index = random.randint(0, len(action_list) - 1)
            fallback_action = action_list[fallback_index]
            
            self.update_exploration_history(current_url, fallback_action, fallback_index)
            
            error_description = f"Error Fallback: {self.format_action_info(fallback_action)} ❌"
            
            self.state_kb.add_page_state(
                page_title=page_title or "Unknown Page",
                action_list=action_list,
                selected_action_index=fallback_index,
                reasoning=f"双模型系统错误: {str(e)}，随机选择动作"
            )
            
            self.thinking_kb.add_thinking(
                prompt="双模型系统错误",
                reasoning=f"Error: {str(e)}",
                action_taken=error_description
            )

            self.history.append(error_description)
            if len(self.history) > self.max_history_length:
                self.history = self.history[-self.max_history_length:]
            
            return fallback_action

    def parse_output(self, output: str, num_actions: int) -> tuple:
        """
        解析LLM输出，提取动作索引和可能的输入文本
        返回一个元组 (动作索引, 输入文本)，如果没有输入文本则为None
        """
        # 内部定义一个函数，用于去除 ANSI 转义序列
        def remove_ansi(text: str) -> str:
            ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            return ansi_pattern.sub('', text)

        cleaned_output = remove_ansi(output).strip()

        # 尝试匹配"索引:文本"格式
        input_pattern = re.compile(r'^(\d+):(.+)$', re.DOTALL)
        match = input_pattern.match(cleaned_output)

        if match:
            index_str, input_text = match.groups()
            try:
                index = int(index_str)
                if 0 <= index < num_actions:
                    return index, input_text.strip()
                else:
                    if self.verbose:
                        print(f"Index {index} out of range [0, {num_actions-1}]")
                    return None, None
            except ValueError:
                if self.verbose:
                    print(f"Cannot convert index: {index_str}")
                return None, None

        # 尝试直接找出数字
        number_match = re.search(r'^\d+', cleaned_output)
        if number_match:
            try:
                index = int(number_match.group())
                if 0 <= index < num_actions:
                    return index, None
                else:
                    if self.verbose:
                        print(f"Index {index} out of range [0, {num_actions-1}]")
                    return None, None
            except ValueError:
                if self.verbose:
                    print(f"Cannot convert index: {number_match.group()}")
                return None, None

        # 尝试从文本中提取最后一个数字作为索引
        numbers = re.findall(r'\d+', cleaned_output)
        if numbers:
            try:
                index = int(numbers[-1])
                if 0 <= index < num_actions:
                    return index, None
                else:
                    if self.verbose:
                        print(f"Index {index} out of range [0, {num_actions-1}]")
                    return None, None
            except ValueError:
                if self.verbose:
                    print(f"Cannot convert index: {numbers[-1]}")
                return None, None

        if self.verbose:
            print(f"Failed to parse output: {cleaned_output[:50]}...")
        return None, None

    def get_exploration_stats(self) -> dict:
        """
        获取探索统计信息
        """
        total_pages = len(self.explored_pages)
        total_unique_actions = sum(len(actions) for actions in self.executed_actions.values())
        
        # 计算重复动作统计
        overused_count = 0
        new_actions_available = 0
        
        for url, actions in self.executed_actions.items():
            for action_sig, count in actions.items():
                if count > 2:  # 固定阈值，之前的max_action_repeats默认值
                    overused_count += 1
        
        # 计算最活跃的页面
        most_active_page = ""
        max_actions = 0
        for url, actions in self.executed_actions.items():
            if len(actions) > max_actions:
                max_actions = len(actions)
                most_active_page = url
        
        return {
            "explored_pages": total_pages,
            "unique_actions_executed": total_unique_actions,
            "overused_actions": overused_count,
            "most_active_page": most_active_page,
            "max_actions_on_page": max_actions,
            "explored_pages_list": list(self.explored_pages)
        }

    def print_exploration_summary(self):
        """
        打印探索摘要
        """
        stats = self.get_exploration_stats()
        print("\n" + "="*50)
        print("🎯 智能探索系统 - 统计摘要")
        print("="*50)
        print(f"📊 已探索页面数量: {stats['explored_pages']}")
        print(f"🎮 执行的唯一动作: {stats['unique_actions_executed']}")
        print(f"⚠️ 过度使用的动作: {stats['overused_actions']}")
        print(f"🏆 最活跃页面: {stats['most_active_page'][:50]}...")
        print(f"🔥 该页面动作数: {stats['max_actions_on_page']}")
        
        if self.bug_indicators:
            print(f"🐛 潜在Bug指标: {len(self.bug_indicators)}")
            for indicator in self.bug_indicators[-3:]:  # 显示最近3个
                print(f"   - {indicator}")
        
        print("="*50)

    def debug_show_final_prompt(self, final_prompt: str):
        """
        调试方法：显示最终发送给LLM的完整prompt结构
        """
        if not self.verbose:
            return
            
        print("\n" + "🔍 DEBUG: 最终Prompt结构" + "="*30)
        
        # 尝试分析prompt的各个部分
        sections = final_prompt.split("\n\n")
        
        for i, section in enumerate(sections[:10]):  # 只显示前10个部分避免过长
            if len(section.strip()) > 0:
                # 识别不同类型的内容
                if "专业知识库" in section:
                    print(f"\n📚 第{i+1}部分 - RAG知识增强:")
                    print("─" * 40)
                    print(section[:300] + "..." if len(section) > 300 else section)
                    
                elif "相关的历史推理经验" in section:
                    print(f"\n🧠 第{i+1}部分 - Thinking知识库:")
                    print("─" * 40)
                    print(section[:300] + "..." if len(section) > 300 else section)
                    
                elif "相似页面状态参考" in section:
                    print(f"\n📊 第{i+1}部分 - 状态知识库:")
                    print("─" * 40)
                    print(section[:300] + "..." if len(section) > 300 else section)
                    
                elif any(keyword in section for keyword in ["可操作的界面元素", "探索策略", "登录步骤"]):
                    print(f"\n🎯 第{i+1}部分 - 基础Prompt:")
                    print("─" * 40)
                    print(section[:500] + "..." if len(section) > 500 else section)
                    
                else:
                    print(f"\n📝 第{i+1}部分 - 其他内容:")
                    print("─" * 40)
                    print(section[:200] + "..." if len(section) > 200 else section)
        
        print(f"\n💾 完整Prompt总长度: {len(final_prompt)} 字符")
        print("🔍 DEBUG: Prompt结构分析完成" + "="*25 + "\n")

    def improve_action_selection_randomness(self, action_list, exploration_scores, current_url: str) -> str:
        """
        已废弃的方法 - 改为纯LLM决策后不再使用
        """
        # 此方法已被删除，改为纯LLM决策
        return ""

    def reset_dual_model_session(self):
        """
        重置双模型会话状态 - 确保测试独立性
        """
        try:
            self.dual_model_system.reset_for_new_test()
            if self.verbose:
                print("Dual model system reset successfully")
        except Exception as e:
            if self.verbose:
                print(f"Failed to reset dual model system: {e}")

    def clear_all_rag_databases(self):
        """
        清空所有RAG数据库（包括新的探索知识库）
        """
        try:
            if self.verbose:
                print("Clearing all RAG databases...")
            self.retriever.clear_vectorstore()
            self.thinking_kb.clear_thinking_vectorstore()
            self.state_kb.clear_state_vectorstore()
            self.exploration_kb.clear_exploration_vectorstore()
            
            if self.verbose:
                print("All RAG databases cleared successfully")
        except Exception as e:
            if self.verbose:
                print(f"Failed to clear RAG databases: {e}")

    def reset_for_new_test(self):
        """
        为新测试重置Agent状态 - 双模型版本
        """
        if self.verbose:
            print("Resetting dual-model agent for new test...")
        
        # 1. 清空历史记录
        self.history = []
        
        # 2. 重置双模型系统状态
        self.reset_dual_model_session()
        
        # 3. 清空所有RAG数据库
        self.clear_all_rag_databases()
        
        if self.verbose:
            stats = self.dual_model_system.get_dual_model_stats()
            print(f"Agent reset complete - R1: {stats['r1_session_id']}, QwQ: {stats['qwq_session_id']}")


def main():
    """🚀 多样性增强双模型协作RAG Agent测试 - R1智能探索 + QwQ多样性决策"""
    # 测试参数
    test_params = {
        "api_key": "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz",
        "embedding_token": "sk-esaaumvchjupuotzcybqofgbiuqbfmhwpvfwiyefacxznnpz",
        "app_name": "Enhanced Dual Model Web Testing",
        "verbose": True,  # 测试时启用详细输出
        "max_tokens": 1024,  # 基础配置，会被各模型覆盖
        "temperature": 0.7,
        "clear_rag_on_init": True,
        "clear_thinking_on_init": True,
        "clear_state_on_init": True,
        "clear_exploration_on_init": True,
        "reset_dual_model_on_init": True,
    }

    print("🚀 多样性增强双模型协作RAG Agent - R1智能探索 + QwQ多样性决策")
    print("=" * 70)
    
    try:
        # 测试Agent初始化
        agent = rag_llm_agent(test_params)
        print("✓ 多样性增强双模型Agent初始化成功")
        
        # 🎲 测试多样性跟踪器功能
        diversity_tracker = agent.dual_model_system.diversity_tracker
        print("\n🎲 多样性跟踪器测试:")
        
        # 测试探索策略选择
        for i in range(5):
            strategy = diversity_tracker.select_exploration_strategy()
            print(f"  🎯 探索策略选择 #{i+1}: {strategy}")
        
        # 测试决策模式选择
        for i in range(5):
            mode = diversity_tracker.select_decision_mode()
            print(f"  ⚡ 决策模式选择 #{i+1}: {mode}")
        
        # 测试多样性分析
        sample_actions = ["点击登录", "输入用户名", "点击搜索", "选择选项", "点击登录", "输入密码"]
        diversity_score = diversity_tracker.calculate_action_diversity(sample_actions)
        print(f"  📊 样本动作多样性分数: {diversity_score:.2f}")
        
        diversity_feedback = diversity_tracker.get_diversity_feedback(sample_actions)
        print(f"  📝 多样性反馈: {diversity_feedback.strip()}")
        
        # 🚀 测试增强查询生成
        print("\n🔍 增强查询生成测试:")
        for strategy in ['conservative', 'innovative', 'balanced', 'risk_focused']:
            queries = diversity_tracker.get_exploration_enhancement_queries("登录页面", strategy)
            print(f"  🎯 {strategy}策略查询: {queries[0][:50]}...")
        
        # ⚡ 测试决策增强上下文
        print("\n⚡ 决策增强上下文测试:")
        for mode in ['conservative', 'exploratory', 'balanced']:
            context = diversity_tracker.get_decision_enhancement_context(mode)
            print(f"  🎯 {mode}模式: {context[:100].replace(chr(10), ' ')}...")
        
        # 测试双模型系统统计（包含多样性信息）
        stats = agent.dual_model_system.get_dual_model_stats()
        print(f"\n✓ 增强双模型系统状态:")
        print(f"  📡 R1会话ID: {stats['r1_session_id']}")
        print(f"  ⚡ QwQ会话ID: {stats['qwq_session_id']}")
        print(f"  📊 探索次数: {stats['total_explorations']}")
        print(f"  💾 缓存状态: {stats['cached_states']}")
        print(f"  🎲 当前多样性分数: {stats['current_diversity_score']:.2f}")
        print(f"  🎯 探索策略使用: {stats['exploration_strategies_usage']}")
        print(f"  ⚡ 决策模式使用: {stats['decision_modes_usage']}")
        
        # 测试重置功能
        original_r1_session = stats['r1_session_id']
        original_qwq_session = stats['qwq_session_id']
        
        agent.reset_for_new_test()
        new_stats = agent.dual_model_system.get_dual_model_stats()
        
        reset_success = (
            original_r1_session != new_stats['r1_session_id'] and
            original_qwq_session != new_stats['qwq_session_id']
        )
        print(f"\n✓ 重置测试: {'Success' if reset_success else 'Failed'}")
        
        # 展示增强架构特性
        print("\n🚀 多样性增强双模型协作架构:")
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  🎲 多样性跟踪器 (DiversityTracker)                        │")
        print("│  ├─ 智能策略选择: 4种探索策略动态平衡                      │")
        print("│  ├─ 多样性决策: 3种决策模式自适应切换                      │")
        print("│  ├─ 实时监控: 动作多样性分数实时计算                       │")
        print("│  └─ 反馈优化: 基于使用频率动态调整权重                     │")
        print("│                                                             │")
        print("│  🔍 增强R1模型 (DeepSeek-R1)                               │")
        print("│  ├─ 策略导向探索: 基于选定策略的深度分析                   │")
        print("│  ├─ 多样化RAG检索: 策略相关的多角度知识获取                │")
        print("│  ├─ 反向学习: 20%概率加入失败案例学习                      │")
        print("│  └─ 输出: 策略化页面分析、多样性测试建议                   │")
        print("│                                                             │")
        print("│  ⚡ 增强QwQ模型 (Qwen/QwQ-32B-Preview)                      │")
        print("│  ├─ 模式导向决策: 基于选定模式的智能选择                   │")
        print("│  ├─ 多样性分析: 动作类型使用频率实时评估                   │")
        print("│  ├─ 历史优化: 优先选择较少使用的动作类型                   │")
        print("│  └─ 输出: 多样性优化的具体动作选择                         │")
        print("└─────────────────────────────────────────────────────────────┘")
        
        print("\n🔄 增强工作流程:")
        print("1. 🎲 多样性评估 → 分析当前测试多样性状态")
        print("2. 🎯 策略选择 → 智能选择R1探索策略 (conservative/innovative/balanced/risk_focused)")
        print("3. 🔍 策略探索 → R1基于策略进行多样化RAG增强深度分析")
        print("4. 💾 知识存储 → 探索结果存储到ExplorationKB")
        print("5. ⚡ 模式决策 → QwQ选择决策模式 (conservative/exploratory/balanced)")
        print("6. 🎨 多样性决策 → 基于多样性分析和历史使用频率智能选择动作")
        print("7. 📊 反馈更新 → 更新多样性指标和策略权重")
        
        print("\n📚 增强四层知识库系统:")
        print("• 🔍 ExplorationKB: R1策略化探索结果 + 多样性洞察")
        print("• 📊 StateKB: 页面状态 + 多样性决策历史")
        print("• 🧠 ThinkingKB: 模型推理 + 失败案例反向学习")
        print("• 📖 RetrieverKB: 专业知识 + 策略导向多样化检索")
        
        print("\n🎯 多样性增强策略:")
        print("🔄 **探索策略 (R1)**:")
        print("  • Conservative: 基于成功经验的稳健探索")
        print("  • Innovative: 寻找未测试区域的创新探索")
        print("  • Balanced: 稳定性与创新性的平衡探索")
        print("  • Risk-focused: 专注边界和风险的导向探索")
        
        print("\n⚡ **决策模式 (QwQ)**:")
        print("  • Conservative: 优选历史成功率高的安全动作")
        print("  • Exploratory: 勇于尝试新路径和未使用动作")
        print("  • Balanced: 稳健验证与创新探索的最佳平衡")
        
        print("\n💡 核心创新亮点:")
        print("🎯 **智能策略选择**: 动态平衡4种探索策略，避免单一模式")
        print("🎲 **多样性驱动**: 实时监控测试多样性，优化动作选择")
        print("📊 **使用频率优化**: 🆕优于📊优于🔥，鼓励尝试新动作类型")
        print("🔄 **自适应权重**: 基于使用频率动态调整策略权重")
        print("🧠 **反向学习**: 从失败案例中学习，避免重复错误")
        print("⚖️ **策略协调**: R1探索策略与QwQ决策模式的智能配合")
        
        print("\n🚀 性能优势:")
        print("💰 **成本效率**: 智能缓存 + 策略化探索大幅降低token消耗")
        print("🎨 **测试覆盖**: 多样性驱动显著提升测试路径覆盖面")
        print("🔍 **深度洞察**: 策略导向的RAG检索提供更精准的专业指导")
        print("⚡ **响应速度**: 模式化决策加速动作选择过程")
        print("🛡️ **系统稳定**: 多层回退机制保障极端情况下的可用性")
        
        print("\n" + "=" * 70)
        print("✅ 多样性增强双模型协作系统测试完成!")
        print("\n🌟 系统特色总结:")
        print("• 🎲 **智能多样性**: 首个集成多样性跟踪的Web测试AI系统")
        print("• 🎯 **策略化探索**: R1模型的4种探索策略自适应选择")
        print("• ⚡ **模式化决策**: QwQ模型的3种决策模式动态平衡")
        print("• 📊 **实时优化**: 基于使用频率的动态权重调整机制")
        print("• 🧠 **经验学习**: 成功案例 + 失败案例的双向学习能力")
        print("• 🚀 **RAG增强**: 四层知识库的策略导向检索系统")
        
        print(f"\n🎉 现在运行 python main.py 体验多样性增强的智能Web测试!")
        print(f"🔥 期待看到 {stats['exploration_strategies_usage']} 策略和 {stats['decision_modes_usage']} 模式的协作效果!")
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 只有在直接运行此文件时才执行测试
    main()
