# CLM num robô: do estado à decisão

Guia para treinar e implantar o CLM (Contrastive Language Model) como o
"System One" de um robô: a cada ciclo, o robô descreve o estado em texto, o CLM
pontua um conjunto de ações candidatas e o robô executa a mais provável.

> **Ideia central.** O CLM não gera texto, ele **escolhe**. Ele embute o estado
> e cada ação candidata (Qwen3-8B + uma cabeça de projeção de 20M parâmetros) e
> escolhe a ação cujo embedding melhor se alinha ao do estado. Por isso ele é
> rápido (~28 ms por estado novo numa RTX 4090, <1 ms em estado repetido) e
> serve bem para controle de alto nível. Ele **não** substitui o controle de
> baixo nível (PID, MPC, cinemática), que continua no robô.

---

## 0. Visão geral da arquitetura

```
 sensores ──► percepção ──► estado em TEXTO ──► CLM (POST /v1/systemone) ──► ação discreta
 (lidar,       (detecção,     "Obstáculo a 0.4 m      Choice sobre ações       │
  câmera,       odometria,     à frente, bateria       fixas ("avançar",        ▼
  IMU...)       mapa)          35%, carga: sim"        "girar_esq", ...)   escudo de segurança
                                                                             │
                                                                             ▼
                                                                    controlador de baixo nível
```

- **Encoder (GPU):** `vllm serve Qwen/Qwen3-8B --runner pooling` na porta 8090.
- **Cabeça CLM (CPU ou GPU):** `clm-serve` na porta 8700, carrega o checkpoint `.pt`.
- **Robô:** cliente leve (`CLMClient`, sem torch) que monta o estado, pergunta e executa.

O CLM só entende **texto** hoje (visão/multimodal está no roadmap). Toda
informação sensorial precisa ser convertida em frases curtas antes de chegar ao
modelo.

---

## 1. Requisitos

| Item | Mínimo |
| --- | --- |
| GPU do encoder | 24 GB (ex.: RTX 4090) com `--max-model-len 2048` |
| Python | ≥ 3.10 |
| Rede robô ↔ servidor | Baixa latência (cabo/Wi-Fi local); o robô só envia texto |
| Disco | ~16 GB para o Qwen3-8B + 75 MB para a cabeça |

Opções de implantação:

1. **Servidor de borda** (recomendado no início): GPU num PC na mesma rede; o
   robô só roda o cliente HTTP.
2. **Embarcado:** placa com GPU suficiente para o Qwen3-8B (ex.: Jetson AGX
   Orin 32/64 GB). Na Jetson, a instalação é por container: veja
   [CONTAINER_JETSON.md](CONTAINER_JETSON.md).

---

## 2. Instalação e teste do modelo base

```bash
git clone <este repositório> && cd CLM
python -m venv .venv && source .venv/bin/activate
pip install -e .            # ou: pip install -r requirements.txt (inclui deps dos exemplos)

# Terminal 1: encoder
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \
     --max-model-len 2048 --port 8090
#   (alternativa com configurações enxutas: GPU=0 ./serve_qwen3_8b.sh)

# Terminal 2: API do CLM (baixa a cabeça de referência na primeira execução)
clm-serve
```

Abra `http://localhost:8700/` (playground) e teste um estado do seu robô com
uma pergunta `choice`. Isso já mostra o desempenho **zero-shot**, antes de
qualquer treino.

Sem GPU por perto, `python tools/playground_mock.py` sobe o playground com um
encoder falso, útil para desenvolver a interface do robô.

### 2.1 Jetson

Na Jetson, **não use `pip install -e .`**. Os torch/vLLM do PyPI não rodam no
driver do JetPack 6. A instalação é por container: veja
[CONTAINER_JETSON.md](CONTAINER_JETSON.md).

---

## 3. Projetar o estado e as ações

Esta é a etapa que mais influencia o resultado.

### 3.1 Ações

- Use um **conjunto fixo e pequeno** de ações discretas de alto nível
  (3 a ~50). Ações fixas são cacheadas (`--action-cache`) e ficam praticamente
  gratuitas.
- Dê a cada ação uma **descrição** em linguagem natural: o CLM embute a
  descrição, não a chave.

```python
ACOES = {
    "avancar":   "Mover para frente em linha reta.",
    "girar_esq": "Girar para a esquerda no lugar.",
    "girar_dir": "Girar para a direita no lugar.",
    "parar":     "Parar imediatamente e aguardar.",
    "recarregar":"Ir até a base de recarga.",
}
```

### 3.2 Estado

- Converta os sensores em **prosa curta e factual** (os heads foram treinados
  com prosa, não JSON). Se passar um `dict`, o servidor o renderiza como linhas
  `chave: valor`.
- Discretize números ("obstáculo a 0.4 m à frente" e não 37 leituras cruas do lidar).
- Inclua só o que importa para a decisão; mantenha bem abaixo de 2048 tokens
  (o excedente é truncado).
- Coloque a pergunta em `instructions`, **não** dentro do estado.
- Estados que se repetem (mesma sala, mesma situação) caem no cache e
  respondem ~2,8× mais rápido; arredondar valores ajuda nisso.

```python
def descrever_estado(s):
    return (
        f"Robô móvel em ambiente interno. "
        f"Obstáculo à frente a {s.dist_frente:.1f} m, à esquerda a {s.dist_esq:.1f} m, "
        f"à direita a {s.dist_dir:.1f} m. "
        f"Bateria {s.bateria:.0f}%. Objetivo: {s.objetivo}."
    )
```

---

## 4. Coletar dados de treino

O CLM aprende pares **(estado, ação correta)**. Fontes possíveis:

1. **Teleoperação:** um humano pilota o robô; registre o estado em texto e a
   ação escolhida a cada ciclo.
2. **Simulador** (Gazebo, Isaac Sim, MuJoCo...) com um planejador "especialista"
   que rotula a melhor ação, como o planejador físico do exemplo
   [`examples/t_rex`](../examples/t_rex/README.md).
3. **Registros do próprio robô** com rótulo de sucesso/falha posterior.

Boas práticas:

- Gere o texto do estado com **a mesma função** `descrever_estado` que vai
  rodar em produção.
- Balanceie as classes (ex.: "parar" costuma ser rara, mas é crítica).
- Separe teste **por episódio/ambiente**, nunca por passo aleatório, para não
  vazar estados quase idênticos entre treino e teste.
- Algumas centenas a poucos milhares de exemplos por ação já costumam ser um
  bom começo, porque a cabeça parte de um modelo pré-treinado.

---

## 5. Formatar o dataset

Há dois modos de treino em `train/finetune.py`. Para um robô com ações fixas,
use **`choice`**.

### 5.1 Modo `choice` (ações fixas, recomendado)

Arquivos Parquet com uma linha por estado e as colunas:

| coluna | conteúdo |
| --- | --- |
| `id` | id único da linha (a validação é separada por id) |
| `state` | texto do estado (ou objeto em JSON) |
| `questions` | JSON no formato System One, ex.: `{"acao": {"type": "choice", ...}}` |
| `gold` | JSON `{"acao": {"label": "girar_esq"}}`; opcionalmente `"probabilities": {...}` para rótulos suaves |
| `workflow` | opcional, nome do subconjunto |

Estrutura de diretório esperada (o script procura `<workflow>/train*.parquet`
e `<workflow>/test*.parquet`; o workflow padrão é `all`):

```
data/robo/
└── all/
    ├── train-00000.parquet
    └── test-00000.parquet
```

> Não passe um único arquivo `.parquet` em `--data`: nesse caso o mesmo arquivo
> é usado como treino **e** teste.

Script de conversão de exemplo (`registros` = lista de `(episodio, passo, estado_texto, acao)`):

```python
import json, os
import pyarrow as pa, pyarrow.parquet as pq

PERGUNTA = {"acao": {"type": "choice",
                     "instructions": "Qual a melhor ação segura para o robô agora?",
                     "criteria": ACOES}}

def linhas(regs):
    for ep, passo, estado, acao in regs:
        yield {"id": f"{ep}-{passo}", "state": estado,
               "questions": json.dumps(PERGUNTA, ensure_ascii=False),
               "gold": json.dumps({"acao": {"label": acao}}),
               "workflow": "all"}

def salvar(regs, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(linhas(regs))), caminho)

episodios_teste = {...}   # separe por episódio
salvar([r for r in registros if r[0] not in episodios_teste], "data/robo/all/train-00000.parquet")
salvar([r for r in registros if r[0] in episodios_teste],     "data/robo/all/test-00000.parquet")
```

### 5.2 Modo `clm` (ações livres/parametrizadas)

Se as ações são texto livre ou variam muito (ex.: "pegar o copo azul na
prateleira 2"), use `--task clm` com um `.jsonl` de transições:

```json
{"state": "Robô na cozinha. Copo azul na prateleira 2...", "action": "pegar copo azul prateleira 2", "task_id": "ep_017", "step_idx": 4}
```

`state` pode ser texto ou uma lista de mensagens de chat. O treino usa InfoNCE
com as outras ações do lote como negativos. `task_id` agrupa episódios para
separar o conjunto de validação (`--holdout-tasks tarefas.json`, uma lista de
`task_id`).

---

## 6. Fine-tuning

### 6.1 O que é fine-tuning

Fine-tuning ("ajuste fino") é pegar um modelo já treinado em muitos dados
gerais e continuar treinando-o, por pouco tempo, com os seus dados
específicos. O modelo não começa do zero: aproveita o que já sabe e só se
adapta à sua tarefa.


No CLM há duas partes:

| parte | tamanho | no fine-tuning |
| --- | --- | --- |
| Encoder Qwen3-8B (texto → vetor) | 8 bilhões de parâmetros | **congelado**, não muda |
| Cabeças de projeção (aproximam estado e ação correta) | ~20 milhões de parâmetros | **treinadas** |

A cabeça de referência (`CLM_v0.1-8B.pt`) foi treinada pelos autores em ~90
milhões de exemplos gerais (perguntas e respostas, negativos difíceis e
trajetórias de agentes), mas nunca viu o seu robô. No comando abaixo:

- `--init-ckpt "$(clm-download)"` começa da cabeça já treinada, e não de pesos
  aleatórios. É isso que faz do treino um fine-tuning.
- `--data data/robo` ajusta a cabeça com os seus pares (estado, ação correta).
- O resultado, `best_head.pt`, é uma cabeça especializada no seu robô.

Por que não treinar do zero:

- **Poucos dados:** centenas ou poucos milhares de exemplos, em vez de milhões.
- **Rápido e barato:** só os ~20M parâmetros da cabeça mudam, e os vetores do
  texto são calculados uma vez e cacheados em `<out-dir>/embeddings`; o treino
  leva minutos, não semanas.
- **Melhor resultado:** o modelo já entende linguagem e decisões; só precisa
  aprender as particularidades da sua tarefa.

**Zero-shot × fine-tuning:** zero-shot é usar o modelo como ele vem, sem
treino (passo 2). Fine-tuning é ajustá-lo com seus dados (este passo). A
diferença de acurácia entre os dois mostra quanto o ajuste ajudou.

### 6.2 Rodando o treino

```bash
# encoder rodando na :8090 (passo 2); reaproveita o servidor em vez de abrir outro vLLM
python train/finetune.py --task choice \
    --data data/robo --workflow all \
    --init-ckpt "$(clm-download)" \
    --embed-url http://127.0.0.1:8090/v1/embeddings --served-model-name qwen3-8b \
    --out-dir runs/robo_v1
```

Parâmetros úteis:

| flag | uso |
| --- | --- |
| `--init-ckpt "$(clm-download)"` | parte da cabeça de referência (sempre recomendado) |
| `--targets soft\|hard` | use `soft` se tiver `probabilities` no `gold` |
| `--loss infonce\|softce` | `softce` = softmax só sobre as opções de cada pergunta; costuma combinar com poucas ações fixas |
| `--epochs`, `--patience`, `--lr`, `--batch` | padrão: 20 épocas, paciência 5, lr 5e-4, lote 256 |
| `--max-len` | padrão 2048; precisa casar com o `--max-model-len` do vLLM |
| `--val-frac` | fração das linhas de treino usada como validação (padrão 0.1) |

Saídas em `runs/robo_v1/`: `best_head.pt` (melhor validação), `final_head.pt`
e `finetune_summary.json` com a acurácia de validação e teste. Compare com a
linha de base majoritária que o script imprime e com o zero-shot do passo 2.

Para iterar em hiperparâmetros de forma sistemática, veja
[FINETUNING.md](FINETUNING.md).

---

## 7. Servir o modelo treinado

```bash
clm-serve --ckpt runs/robo_v1/best_head.pt          # serve como clm-latest
# ou várias versões lado a lado:
clm-serve --model robo-v1=runs/robo_v1/best_head.pt --model robo-v2=runs/robo_v2/best_head.pt
```

O checkpoint é recarregado automaticamente quando o arquivo muda (não precisa
reiniciar após um novo treino). Em rede compartilhada, defina `CLM_API_KEY`.
Use `--no-ui` em produção.

---

## 8. Laço de decisão no robô

```python
import time
from clm import CLMClient, Choice

cliente = CLMClient(base_url="http://<ip-do-servidor>:8700",  # ou CLM_BASE_URL
                    timeout=0.2)   # o padrão é 300 s; num robô, falhe rápido e pare
PERGUNTA = {"acao": Choice(instructions="Qual a melhor ação segura para o robô agora?",
                           criteria=ACOES)}
CONF_MINIMA = 0.30
PERIODO = 0.1   # 10 Hz

while True:
    t0 = time.monotonic()
    s = ler_sensores()
    try:
        r = cliente.system_one(state=descrever_estado(s), questions=PERGUNTA)
        resp = r.answers["acao"]
        acao, probs = resp.choice, resp.probabilities
        if resp.confidence < CONF_MINIMA:
            acao = "parar"                                  # incerto: comportamento seguro
    except Exception:
        acao = "parar"                                      # servidor fora / timeout

    seguras = acoes_seguras(s)                              # regras físicas, independentes do modelo
    if acao not in seguras:                                 # escudo de segurança
        acao = max(seguras, key=probs.get) if seguras else "parar"

    executar(acao)
    time.sleep(max(0.0, PERIODO - (time.monotonic() - t0)))
```

Para rodar no mesmo processo, sem HTTP, use `Engine` (exige torch):

```python
from clm import Engine
engine = Engine(emb_url="http://127.0.0.1:8090/v1/embeddings")
resp = engine.answer(descrever_estado(s), PERGUNTA)["answers"]["acao"]
```

### Segurança (obrigatório)

- **Escudo:** como no exemplo T-Rex, uma camada de regras rejeita ações
  inseguras e escolhe a ação segura mais provável segundo o modelo.
- **Parada de emergência** em hardware, independente de software.
- **Watchdog:** se uma resposta chegar atrasada (estado já mudou), descarte-a.
- **Fallback** para "parar" quando o servidor cai ou a confiança é baixa.
- Registre `probabilities`, `confidence`, `r.latency_ms` e as intervenções do
  escudo. Esses registros viram dados para o próximo fine-tuning.

---

## 9. Avaliação e iteração

1. **Offline:** acurácia em `test` (no `finetune_summary.json`) e matriz de
   confusão por ação.
2. **Simulação:** taxa de sucesso, colisões e quantas vezes o escudo interveio
   (uma boa política quase nunca precisa do escudo).
3. **Robô real, em ambiente controlado:** velocidade reduzida, área isolada.
4. **Latência:** meça ponta a ponta (sensor → atuação). Estado novo ≈ 30 ms no
   servidor mais a rede; o período do laço precisa acomodar isso.
5. **Ciclo de dados:** colete os casos em que o escudo interveio ou a confiança
   foi baixa, rotule, adicione ao dataset e retreine (o `clm-serve` recarrega o
   novo `best_head.pt` sozinho).

---

## 10. Checklist

- [ ] Encoder vLLM + `clm-serve` rodando e respondendo no playground
- [ ] Conjunto de ações fixas com descrições claras
- [ ] Função `descrever_estado` única para coleta e produção
- [ ] Dataset em `data/robo/all/{train,test}-*.parquet`, separado por episódio
- [ ] Teste zero-shot registrado como linha de base
- [ ] Fine-tuning com `--init-ckpt "$(clm-download)"` → `best_head.pt`
- [ ] `clm-serve --ckpt runs/.../best_head.pt`
- [ ] Laço no robô com escudo, watchdog, fallback e parada de emergência
- [ ] Avaliação em simulação antes do robô real
- [ ] Registro de decisões para o próximo ciclo de treino
