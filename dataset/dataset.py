import torch
import random
from dataset import SeqDataset, NonSeqDataset
from utils import neg_sample
from typing import Tuple

class SeqNegPreDataset(SeqDataset):
    def __init__(
        self, 
        user_seq: list, 
        long_sequence: list,
        **kwargs
    ) -> None:
        # user_seq, long_sequence 구한 것을 받는 형태로 작동
        super().__init__(**kwargs)
        self.user_seq = user_seq # 유저별 본 영화 시퀀스
        self.long_sequence = long_sequence # data상 모든 영화 나열
        self.part_sequence = [] # 시퀀스를 계단식으로 쪼갤것임
        self.split_sequence()

    def split_sequence(self) -> None:
        for seq in self.user_seq:
            input_ids = seq[-(self.max_seq_len + 2) : -2]  # keeping same as train set
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[: i + 1])
    
    def __len__(self) -> int:
        return len(self.part_sequence)
        
    def __getitem__(self, index) -> Tuple[torch.tensor, ...]:
        sequence = self.part_sequence[index]  # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random() # 0~1 사이의 난수 생성
            if prob < self.config_dataset["mask_p"]: # neg_items 선정
                masked_item_sequence.append(self.config["mask_id"])
                neg_items.append(neg_sample(item_set, self.config["item_size"]))
            else: # 그 아이템을 neg_items에 넣음
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.config["mask_id"])
        neg_items.append(neg_sample(item_set, self.config["item_size"]))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id : start_id + sample_length]
            neg_segment = self.long_sequence[
                neg_start_id : neg_start_id + sample_length
            ]
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.config["mask_id"]] * sample_length
                + sequence[start_id + sample_length :]
            )
            pos_segment = (
                [self.config["mask_id"]] * start_id
                + pos_segment
                + [self.config["mask_id"]] * (len(sequence) - (start_id + sample_length))
            )
            neg_segment = (
                [self.config["mask_id"]] * start_id
                + neg_segment
                + [self.config["mask_id"]] * (len(sequence) - (start_id + sample_length))
            )

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_seq_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_seq_len :]
        pos_items = pos_items[-self.max_seq_len :]
        neg_items = neg_items[-self.max_seq_len :]

        masked_segment_sequence = masked_segment_sequence[-self.max_seq_len :]
        pos_segment = pos_segment[-self.max_seq_len :]
        neg_segment = neg_segment[-self.max_seq_len :]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attributes = []
        for item in pos_items:
            attribute = [0] * self.config["attribute_size"]
            try:
                now_attribute = self.config["item2attribute"][str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)

        assert len(attributes) == self.max_seq_len
        assert len(masked_item_sequence) == self.max_seq_len
        assert len(pos_items) == self.max_seq_len
        assert len(neg_items) == self.max_seq_len
        assert len(masked_segment_sequence) == self.max_seq_len
        assert len(pos_segment) == self.max_seq_len
        assert len(neg_segment) == self.max_seq_len

        cur_tensors = (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )
        return cur_tensors


class SeqNegDataset(SeqDataset):
    def __init__(
        self, 
        user_seq, 
        data_type="train",
        test_neg_items = None,
        **kwargs
        ) -> None:
        super.__init__(**kwargs)
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type

    def __len__(self) -> int:
        return len(self.user_seq)

    def __getitem__(self, index) -> Tuple[torch.tensor, ...]:
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test", "submission"}

        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        elif self.data_type == "test":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))
        
        pad_len = self.max_seq_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_seq_len :]
        target_pos = target_pos[-self.max_seq_len :]
        target_neg = target_neg[-self.max_seq_len :]

        assert len(input_ids) == self.max_seq_len
        assert len(target_pos) == self.max_seq_len
        assert len(target_neg) == self.max_seq_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors