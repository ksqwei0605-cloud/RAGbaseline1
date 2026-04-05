# RAGbaseline1

baseline_v1/
├── data/
│   ├── your_video.pkl
│   └── clip_docs.json
├── cache/
│   └── clip_embeddings.pkl
├── scripts/
│   ├── inspect_mem.py
│   ├── build_clip_docs.py
│   ├── retrieval.py
│   └── qa_once.py
└── outputs/
    └── qa_logs.json
#
inspect_mem.py

目的：检查 pickle 文件的内部结构（可选，用于理解数据）


```bash
python /home/ok/m3-agent/baseline1/scripts/inspect_graph.py \   --pkl /home/ok/m3-agent/data/memory_graphs/robot/bedroom_01.pkl
```

  
输出例子：
```bash
[1] Graph object type: <class 'mmagent.videograph.VideoGraph'>

[2] Field existence
  - nodes: YES
  - text_nodes: YES
  - text_nodes_by_clip: YES
  - edges: YES
  - event_sequence_by_clip: YES

[3] Basic counts
  - nodes total: 2015
  - text_nodes total: 1995
  - edges total: 2154
  - clips in text_nodes_by_clip: 73
  - clips in event_sequence_by_clip: 73
```


#
build_clip_docs.py

目的：从 pickle 提取文本内容和预计算的 embeddings，生成 JSON

```bash
python baseline1/scripts/build_clip_docs.py --pkl data/memory_graphs/robot/bedroom_01.pkl --out-dir baseline1/outputs
```

json中例子：
```bash
"clip_id": 0,
    "node_ids": [
      0,
      1,
      2,
      3,
      4,
    
    ],
    "num_text_nodes": 24,
    "text": "The video begins in a kitchen area, showing a stainless steel refrigerator, a dining table with a blue tablecloth, and a shelving unit.\nThe person enters the room, holding a phone and wearing a black jacket, black pants, white socks, and a black 'COLORADO' cap.\nThe person removes the black jacket, revealing a white long-sleeved shirt underneath.\nThe person hangs the black jacket and a gold sequined bag on a white coat rack near a window.\nThe person picks up a brown studded bag from a blue blanket on the floor.\nThe person puts on ",
    "embedding": [
      -0.03476526215672493,
```
#
retrieval.py

目的：根据问题检索最相关的 top-k clips（三种后端）

```bash
python /home/ok/m3-agent/baseline1/scripts/retrieval.py \
  --clip-docs /home/ok/m3-agent/baseline1/outputs/clip_docs_with_embeddings.json \
  --question "What happened in the kitchen?" \
  --backend precomputed \
  --top-k 3
```

输出例子：
```bash
================================================================================ Question: What happened in the kitchen? ​ ​   Backend: precomputed ================================================================================ ​ ​   [1] clip_id=41 score=-0.036282 num_text_nodes=28 ​     <voice_24> and <voice_459> are seated at a table covered with a blue and yellow checkered tablecloth with a floral design. The table is set with a tray of French fries, a slice of cake with a strawberry decoration, a ... ​ ​  
  [2] clip_id=39 score=-0.037856 num_text_nodes=29 ​     <voice_459> and another individual are seated at a table covered with a blue tablecloth with a yellow floral design. The table is set with a takeout container of French fries and a slice of cake with a strawberry deco... ​ ​   
  [3] clip_id=63 score=-0.038251 num_text_nodes=29 ​     <voice_459> and another individual are seated at a table covered with a blue tablecloth with a yellow floral design. The table is set with a white mug, a small cake decorated with a strawberry, a container of French f...
```


#
qa_once.py
目的：检索 → 构建上下文 → 调用 chat API → 生成答案

```bash
python baseline1/scripts/qa_once.py \
  --clip-docs /home/ok/m3-agent/baseline1/outputs/clip_docs_with_embeddings.json \
  --question "What happened in the kitchen?" \
  --backend precomputed \
  --top-k 3 \
  --api-key 3610b0b0-c7ef-4a3c-a03d-51f1dfe5a480 \
  --base-url "https://ark.cn-beijing.volces.com/api/v3" \
  --chat-model doubao-seed-2-0-lite-260215
```

输出例子：
```bash
================================================================================
One-shot RAG QA Result
================================================================================
Question: What happened in the kitchen?
Retrieval backend: precomputed
Top-k: 3

Retrieved clips:
[1] clip_id=41 score=-0.036282 num_text_nodes=28
    preview: <voice_24> and <voice_459> are seated at a table covered with a blue and yellow checkered tablecloth with a floral design. The table is set with a tray of French fries, a slice of cake with a strawberry decoration, a white mug, a black pouch, pens, a bottle of orange soda, a bottle of water, and some papers. <voice_...
[2] clip_id=39 score=-0.037856 num_text_nodes=29
    preview: <voice_459> and another individual are seated at a table covered with a blue tablecloth with a yellow floral design. The table is set with a takeout container of French fries and a slice of cake with a strawberry decoration. A white mug, a black bag, pens, a bottle of soda, and a bottle of water are also on the tabl...
[3] clip_id=63 score=-0.038251 num_text_nodes=29
    preview: <voice_459> and another individual are seated at a table covered with a blue tablecloth with a yellow floral design. The table is set with a white mug, a small cake decorated with a strawberry, a container of French fries, a plate with more French fries and a sandwich, a bottle of orange juice, a plastic water bottl...

Final answer:
This space (likely a kitchen/dining area of a private residence) hosted a casual supportive mentorship conversation over an informal shared meal between two people:
- The mentee <voice_459> shared the challenges of programming, his ambition to make a social impact through his future programming career, his self-doubt about achieving his dream, expressed appreciation for the mentor's guidance (on both technical skills and personal growth), discussed motivational themes from *The Old Man and the Sea* and music, and asked how to balance pressure.
- The mentor consistently offered encouragement, emotional validation and support to the mentee, forming a warm, hopeful, supportive interaction.
================================================================================
```

#
run_baseline1_eval
读取robot.json中的指定pkl里面的question和answer，然后把其中的question问问题给模型生成回答，把这个回答和相应的问题和标准答案生成一个json
```bash
python /home/ok/m3-agent/baseline1/scripts/run_baseline1_eval.py \
  --annotation_file /home/ok/m3-agent/data/annotations/robot.json \
  --target_pkl /home/ok/m3-agent/data/memory_graphs/robot/bedroom_01.pkl \
  --clip_docs_dir /home/ok/m3-agent/baseline1/outputs \
  --output_file /home/ok/m3-agent/baseline1/outputs/eval_results.json \
  --api_key "3610b0b0-c7ef-4a3c-a03d-51f1dfe5a480" \
  --base_url "https://ark.cn-beijing.volces.com/api/v3" \
  --chat_model "doubao-seed-2-0-lite-260215" \
  --top_k 3 \
  --backend precomputed
```
输出例子：
```bash
[
  {
    "mem_path": "data/memory_graphs/robot/living_room_06.pkl",
    "question": "What are the drinks for the three people?",
    "ground_truth_answer": "Coffee.",
    "model_answer": "Based on the memory context, the drinks for the three people are coffee.",
    "status": "ok",
    "id": "living_room_06_Q01",
    "reasoning": "...",
    "timestamp": "00:43",
    "type": ["Cross-Modal Reasoning"]
  },
  ...
]
```

#
judge_baseline1_with_seed
用外部api来根据刚才生成的那个json判断回答是否正确
```bash
python /home/ok/m3-agent/baseline1/scripts/judge_baseline1_with_seed.py \
  --input_file /home/ok/m3-agent/baseline1/outputs/eval_results.json \
  --output_file /home/ok/m3-agent/baseline1/outputs/eval_results_judged.json \
  --api_key "3610b0b0-c7ef-4a3c-a03d-51f1dfe5a480" \
  --base_url "https://ark.cn-beijing.volces.com/api/v3" \
  --model "doubao-seed-2-0-lite-260215"
```

输出例子：
```bash
[1/15] Judging sample: bedroom_01_Q01
[2/15] Judging sample: bedroom_01_Q02
[3/15] Judging sample: bedroom_01_Q03
[4/15] Judging sample: bedroom_01_Q04
[5/15] Judging sample: bedroom_01_Q05
[6/15] Judging sample: bedroom_01_Q06
[7/15] Judging sample: bedroom_01_Q07
[8/15] Judging sample: bedroom_01_Q08
[9/15] Judging sample: bedroom_01_Q09
[10/15] Judging sample: bedroom_01_Q10
[11/15] Judging sample: bedroom_01_Q11
[12/15] Judging sample: bedroom_01_Q12
[13/15] Judging sample: bedroom_01_Q13
[14/15] Judging sample: bedroom_01_Q14
[15/15] Judging sample: bedroom_01_Q15
Total samples: 15
Judged samples: 15
Correct samples: 1
Accuracy: 0.0667
Saved judged results to: /home/ok/m3-agent/baseline1/outputs/eval_results_judged.json
```

```bash
 {

    "mem_path": "data/memory_graphs/robot/bedroom_01.pkl",

    "question": "Which coat rack should Emma's coat be laced, taller one or shorter one?",

    "ground_truth_answer": "Taller coat rack",

    "model_answer": "This information is not mentioned or available in the provided context. There is no reference to Emma, Emma's coat, or multiple (taller/shorter) coat racks in the retrieved memory snippets.",

    "status": "ok",

    "id": "bedroom_01_Q01",

    "reasoning": "Lily进门把外套挂在了高一点的架子上，Emma的外套应该也挂在高一点的架子",

    "timestamp": "07:20",

    "type": [

      "Person Understanding"

    ],

    "judge_result": "No",

    "is_correct": false

  },
```

问题：
wsl与git用法，本地仓库和远程仓库保存？
试试提高正确率，多找几个top?
分离事件记忆和语义记忆
接其他seed模型比如Pro
匹配向量相似度时使用外部api是否会更好？tfidf可以直接删除这一模块
试试多用几个pkl一起测评
完成单层rag内容baseline1
