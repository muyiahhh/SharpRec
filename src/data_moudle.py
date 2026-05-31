import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any

class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length
        
        print(f"Loading dataset from {data_path}...")
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        # 适配你的数据格式 {"messages": [...]}
                        if "messages" in item:
                            self.data.append(item["messages"])
                    except Exception as e:
                        print(f"Skipping invalid json line: {e}")
            print(f"Successfully loaded {len(self.data)} samples.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]
        return self.preprocess_llama2(messages)

    def preprocess_llama2(self, messages: List[Dict[str, str]]):
        """
        处理 Llama 2 格式的对话数据。
        Input Format: {"messages": [{"role": "system",...}, {"role": "user",...}, ...]}
        Target Format: <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST] {assistant} </s>
        """
        system_content = ""
        user_content = ""
        assistant_content = ""

        # 提取各个角色的内容
        # 注意：这里假设每条数据只有一轮对话（System -> User -> Assistant）
        # 如果有多轮对话，逻辑需要更复杂（拼接历史），这里按单轮优化
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content

        # 1. 构建 Prompt 部分 (输入)
        # Llama 2 的 system prompt 需要包裹在 <<SYS>> 标签中
        if system_content:
            prompt_text = f"[INST] <<SYS>>\n{system_content}\n<</SYS>>\n\n{user_content} [/INST] "
        else:
            prompt_text = f"[INST] {user_content} [/INST] "

        # 2. 构建 Output 部分 (输出)
        response_text = f"{assistant_content} {self.tokenizer.eos_token}"

        # 3. Tokenize
        # add_special_tokens=True 会在开头自动加 <s> (BOS token)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True) 
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)

        # 4. 拼接 input_ids
        input_ids = prompt_ids + response_ids
        
        # 5. 构建 Labels
        # Prompt 部分 (System + User) 不计算 Loss，设为 -100
        # Response 部分 (Assistant) 计算 Loss，保留原 token id
        labels = [-100] * len(prompt_ids) + response_ids

        # 6. 截断处理
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

class DataCollatorForSAFT:
    """
    负责将 batch 内的数据动态 Padding 到相同长度
    """
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        # 确保 pad_token_id 存在，Llama 通常没有 pad token，使用 eos 或 unk 代替
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        # 创建 Attention Mask (非 padding 部分为 1)
        attention_mask = input_ids_padded.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask
        }