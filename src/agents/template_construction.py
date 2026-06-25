import os
import json
import random
from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer, util
from submodlib import LogDeterminantFunction
import numpy as np
from Levenshtein import distance as levenshtein


class BaseTemplateConstruction(ABC):
    def __init__(self, question: str, dataset: str):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.question = question
        self.dataset = dataset

    @abstractmethod
    def few_shot_with_dpp(self) -> tuple[str, List[str]]:
        pass

    @abstractmethod
    def full_shot_with_diversity(self) -> tuple[str, List[str]]:
        pass

    @abstractmethod
    def static_prompt_construction(self) -> tuple[str, List[str]]:
        pass

    def load_dataset_for_few_shot(self, path: str) -> List[str]:
        questions = []
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        for x in data:
            if x.get("Question") is not None:
                questions.append(x.get("Question"))
        return questions

    def cos_sim(self, element: str, model: SentenceTransformer, labels_sim: np.ndarray, threshold: int = 2) -> np.ndarray:
        x = model.encode([element])
        res = util.dot_score(x, labels_sim)
        res = res.squeeze()
        y = np.array(res)
        ind = np.argpartition(y, -threshold)[-threshold:]
        ind = ind[np.argsort(y[ind])]
        return ind

    def cos_sim_least(self, element: str, model: SentenceTransformer, labels_sim: np.ndarray, threshold: int = 2, most_similar: bool = False) -> np.ndarray:
        x = model.encode([element])
        res = util.dot_score(x, labels_sim)
        res = res.squeeze()
        y = np.array(res)
        if most_similar:
            ind = np.argpartition(y, -threshold)[-threshold:]
        else:
            ind = np.argpartition(y, threshold)[:threshold]
        ind = ind[np.argsort(y[ind])]
        return ind

    def most_similar_items(self, question: str, questions: List[str], threshold: int = 1) -> List[str]:
        labels_sim = self.model.encode(questions)
        indexes = self.cos_sim(question, self.model, labels_sim, threshold=threshold)
        if len(indexes) == 1:
            res_list = [indexes[0]]
        else:
            res_list = [[i] for i in indexes]
        return res_list
    
    def most_similar_items_dynamic(self, question: str, questions: List[str], threshold: int = 1) -> List[str]:
        labels_sim = self.model.encode(questions)
        indexes = self.cos_sim(question, self.model, labels_sim, threshold=threshold)
        if len(indexes) == 1:
            res_list = questions[indexes[0]]
        else:
            res_list = [questions[i] for i in indexes]
        return res_list

    def dpp_function(self, query_similarities: np.ndarray, diversity_matrix: np.ndarray) -> List[int]:
        # query_similarities: shape (n,) - similarity of each candidate to query
        # diversity_matrix: shape (n, n) - pairwise diversity between candidates
        # Combine quality (query similarity) and diversity into kernel matrix
        # K_ij = q_i * diversity_ij * q_j
        q_col = query_similarities.reshape(-1, 1)
        q_row = query_similarities.reshape(1, -1)
        K = q_col * diversity_matrix * q_row

        n = len(query_similarities)
        obj_log_det = LogDeterminantFunction(n=n, mode="dense", lambdaVal=0, sijs=K)
        greedy_indices_and_scores = obj_log_det.maximize(
            budget=3,
            optimizer='NaiveGreedy',
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            verbose=False
        )
        greedy_indices, greedy_scores = zip(*greedy_indices_and_scores)
        return list(greedy_indices)


class TemplateConstruction(BaseTemplateConstruction):
    def __init__(self, question: str, dataset: str, cos: bool = False, div: bool = False):
        super().__init__(question, dataset)

        self.cos = cos
        self.div = div

    def few_shot_with_dpp(self) -> tuple[str, List[str]]:
        path = "/home/infai/Desktop/UI_SysBio_Dec03_2025/UI_SYSBIO_Sol_Lib"
        questions = self.load_dataset_for_few_shot(f"{path}/{self.dataset}")

        # Compute similarity between query question and all candidate questions
        query_embedding = self.model.encode([self.question])
        candidate_embeddings = self.model.encode(questions)
        query_similarities = util.dot_score(query_embedding, candidate_embeddings).squeeze().cpu().numpy()

        with open(f"{path}/{self.dataset}", "r", encoding="utf-8") as file:
            data = json.load(file)
            action_sequence_list = [x.get("Action_Sequence") for x in data]

            # Compute diversity matrix based on action sequence similarity
            diversity_matrix = []
            for i in action_sequence_list:
                x = [1 - levenshtein(i, j) / max(len(i), len(j)) for j in action_sequence_list]
                diversity_matrix.append(x)
            diversity_matrix = np.array(diversity_matrix)

        indices = self.dpp_function(query_similarities, diversity_matrix)
        final_template = (
            f"Example 1:\n{data[indices[0]].get('One_Shot')}\n\n"
            f"Example 2:\n{data[indices[1]].get('One_Shot')}\n\n"
            f"Example 3:\n{data[indices[2]].get('One_Shot')}"
        )
        selected_questions = [
            data[indices[0]].get('Question'),
            data[indices[1]].get('Question'),
            data[indices[2]].get('Question')
        ]
        return final_template, selected_questions

    def full_shot_with_diversity(self) -> tuple[str, List[str]]:
        path = "/home/infai/Desktop/UI_SysBio_Dec03_2025/UI_SYSBIO_Sol_Lib"
        questions = self.load_dataset_for_few_shot(f"{path}/{self.dataset}")
        fetching_ques = self.most_similar_items_dynamic(self.question, questions)
        with open(f"{path}/{self.dataset}", "r", encoding="utf-8") as file:
            data = json.load(file)
            action_sequence = ""
            action_sequence_list = []
            final_template = ""
            selected_questions = []
            for idx, x in enumerate(data):
                if x.get("Question").strip() == fetching_ques.strip():
                    action_sequence = x.get("Action_Sequence").strip()
                    final_template = f"Example 1:\n{final_template}{x.get('One_Shot')}"
                    selected_questions.append(x.get('Question'))
                action_sequence_list.append(x.get("Action_Sequence").strip())
            similar_sequences = self.model.encode(action_sequence_list)
            if self.cos:
                indexes = self.cos_sim_least(action_sequence, self.model, similar_sequences, 4, most_similar=True)
                # Skip indexes[0] as it's the same as Example 1, use indexes[1] and indexes[2]
                final_template += f"\n\nExample 2:\n{data[indexes[1]].get('One_Shot')}\n\nExample 3:\n{data[indexes[2]].get('One_Shot')}"
                selected_questions.extend([data[indexes[1]].get('Question'), data[indexes[2]].get('Question')])
            elif self.div:
                indexes = self.cos_sim_least(action_sequence, self.model, similar_sequences, 10, most_similar=False)
                selected_indices = random.sample(list(indexes), 3)
                final_template += f"\n\nExample 2:\n{data[selected_indices[0]].get('One_Shot')}\n\nExample 3:\n{data[selected_indices[1]].get('One_Shot')}"
                selected_questions.extend([data[selected_indices[0]].get('Question'), data[selected_indices[1]].get('Question')])
            else:
                indexes = self.cos_sim_least(action_sequence, self.model, similar_sequences, 2)
                final_template += f"\n\nExample 2:\n{data[indexes[0]].get('One_Shot')}\n\nExample 3:\n{data[indexes[1]].get('One_Shot')}"
                selected_questions.extend([data[indexes[0]].get('Question'), data[indexes[1]].get('Question')])
            return final_template, selected_questions

    def static_prompt_construction(self) -> tuple[str, List[str]]:
        path = os.getcwd()
        with open(f"{path}/{self.dataset}", "r", encoding="utf-8") as file:
            data = json.load(file)
            selected_examples = random.sample(data, 3)
            final_template = ""
            selected_questions = []
            for idx, x in enumerate(selected_examples, 1):
                final_template += f"\n\nExample {idx}:\n{x.get('One_Shot')}"
                selected_questions.append(x.get('Question'))
            return final_template, selected_questions