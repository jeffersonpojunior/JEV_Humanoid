# CLM numa Jetson via container

Guia para rodar o CLM embarcado numa NVIDIA Jetson Orin com JetPack 6: o
encoder Qwen3-8B (vLLM, porta 8090) e a API do CLM (`clm-serve`, porta 8700),
os dois num container Docker.

Testado numa Jetson Orin 32 GB com **JetPack 6.2 (L4T R36.4.3)**, Ubuntu
22.04, imagem base `dustynv/vllm:0.8.6-r36.4-cu128-24.04` (torch 2.7.0,
CUDA 12.8, vLLM 0.8.6).

- [Por que container](#por-que-container)
- [Instalação inicial (uma vez por robô)](#instalação-inicial-uma-vez-por-robô)
- [Uso no dia a dia](#uso-no-dia-a-dia)
- [Configuração](#configuração)
- [Vários robôs](#vários-robôs)
- [Problemas comuns](#problemas-comuns)

---

## Por que container

**Não use `pip install -e .` direto na Jetson.** O PyPI instala torch e vLLM
compilados para CUDA 12.8+/13 e para ARM de servidor (SBSA). Na Orin, o driver
vem do JetPack e vai só até CUDA 12.6 (JetPack 6), e o resultado é:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12060)
```

Atualizar o driver não é possível nem necessário. Também não adianta trocar
para o índice `cu126` do PyTorch: as wheels aarch64 de lá continuam sendo SBSA.
O que resolve é usar torch e vLLM compilados para Jetson, que vêm prontos nas
imagens do [jetson-containers](https://github.com/dusty-nv/jetson-containers).
A imagem do CLM é construída sobre uma delas:

| Arquivo | Função |
| --- | --- |
| `docker/Dockerfile.jetson` | Imagem do CLM sobre a imagem de vLLM para Jetson. Instala o CLM com `--no-deps` e fixa o torch/vLLM da base |
| `docker/build_jetson.sh` | Constrói a imagem. Por padrão usa a base do JetPack 6.2; com `AUTOTAG=1`, escolhe a base compatível com o seu L4T (`autotag vllm`) |
| `docker/docker-compose.jetson.yml` | Sobe o encoder (:8090) e o `clm-serve` (:8700), com reinício automático |

---

## Instalação inicial (uma vez por robô)

Rode todos os comandos **fora de qualquer venv** (`deactivate` antes).

### Passo 1: conferir o JetPack e o disco

```bash
cat /etc/nv_tegra_release     # esperado: R36 (release), REVISION: 4.x  → JetPack 6.x
df -h / ~                     # precisa de ~30 GB livres
```

São cerca de 6 GB para a imagem base, 16 GB para o Qwen3-8B e mais alguns GB
de folga. Se o disco principal for pequeno e houver um NVMe, mova os dados do
Docker para ele: acrescente `"data-root": "/caminho/no/nvme/docker"` ao
`/etc/docker/daemon.json` (depois do passo 2) e reinicie o Docker.

### Passo 2: instalar o Docker e o runtime NVIDIA

```bash
sudo apt update
sudo apt install -y nvidia-container curl
curl -fsSL https://get.docker.com | sh
sudo systemctl --now enable docker

# registrar o runtime nvidia e torná-lo o padrão
sudo nvidia-ctk runtime configure --runtime=docker
sudo python3 - <<'EOF'
import json
p = "/etc/docker/daemon.json"
d = json.load(open(p))
d["default-runtime"] = "nvidia"
json.dump(d, open(p, "w"), indent=4)
EOF
sudo systemctl restart docker

# usar docker sem sudo
sudo usermod -aG docker $USER
```

Depois do `usermod`, **saia e entre de novo** (por SSH, reconecte). Confira:

```bash
groups | grep docker
docker info | grep -i runtime   # deve listar "nvidia" e "Default Runtime: nvidia"
docker compose version
```

Se o `apt` ficar em `Could not get lock /var/lib/dpkg/lock-frontend`, outro
`apt` está rodando (em geral a atualização automática do Ubuntu). Espere ele
terminar e não mate o processo.

### Passo 3: trazer o repositório para o robô

```bash
cd ~/Workspace
git clone <este repositório> CLM     # ou, se já existe: cd CLM && git pull
```

### Passo 4: construir a imagem do CLM

```bash
cd ~/Workspace/CLM
./docker/build_jetson.sh
```

A primeira vez baixa a imagem base (5.7 GB). No fim do build aparece uma linha
como `torch 2.7.0 | vllm 0.8.6 | clm 0.1.0`.

O build não importa o torch. Na Jetson, as bibliotecas da GPU só são montadas
pelo runtime `nvidia` quando o container roda. Por isso, o teste da GPU vem
depois do build:

```bash
docker run --rm --runtime nvidia --network host clm-jetson:latest \
  python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
#   esperado: 2.7.0 12.8 True
```

O `--network host` é necessário: veja [iptables](#problemas-comuns).

### Passo 5: subir os serviços pela primeira vez

```bash
cd ~/Workspace/CLM
docker compose -f docker/docker-compose.jetson.yml up -d
docker compose -f docker/docker-compose.jetson.yml logs -f encoder   # Ctrl+C sai dos logs, não para o serviço
```

Na primeira subida, o encoder baixa ~16 GB do Qwen3-8B para
`~/.cache/huggingface`, que fica fora do container e é reaproveitado. O `clm`
só inicia quando o encoder responde em `/health`, e na primeira vez ele baixa
a cabeça de referência para `~/.cache/clm`. Para testar:

```bash
curl -s localhost:8090/health && echo encoder ok
curl -s localhost:8700/ >/dev/null && echo clm ok    # playground em http://<ip-do-robô>:8700/
```

Teste no playground um estado cuja resposta você já conhece. As embeddings
vêm do vLLM 0.8.6 e não da versão usada no treino, então vale conferir se as
escolhas fazem sentido.

### Opcional: jetson-containers (só para o `autotag`)

Só é necessário para descobrir a base certa em outra versão do JetPack. Na
6.2, a base padrão do Dockerfile já é a correta.

```bash
cd ~/Workspace
git clone https://github.com/dusty-nv/jetson-containers
python3 -m pip install --user -r jetson-containers/requirements.txt
bash jetson-containers/install.sh
cd ~/Workspace/CLM && AUTOTAG=1 ./docker/build_jetson.sh
```

Não rode o `install.sh` com o venv do CLM ativo: as dependências vão para o
venv e o `autotag` (que usa o `python3` do sistema) falha depois.

---

## Uso no dia a dia

Depois da instalação, **não é preciso fazer nada para ligar o CLM**. O Docker
inicia junto com o sistema (`systemctl enable docker`), e os serviços têm
`restart: unless-stopped`. Quando o robô liga, o encoder e o `clm` sobem
sozinhos. O encoder leva alguns minutos para carregar o modelo.

Para não repetir o `-f`, adicione ao `~/.bashrc` do robô:

```bash
export COMPOSE_FILE=~/Workspace/CLM/docker/docker-compose.jetson.yml
```

Com isso, os comandos abaixo funcionam de qualquer pasta. Sem a variável, use
`docker compose -f docker/docker-compose.jetson.yml ...` a partir de
`~/Workspace/CLM`.

| Tarefa | Comando |
| --- | --- |
| Ver se está rodando | `docker compose ps` |
| Conferir se responde | `curl -s localhost:8090/health && curl -s localhost:8700/ >/dev/null && echo ok` |
| Ver logs | `docker compose logs -f encoder` ou `docker compose logs -f clm` |
| Ligar (se foi parado) | `docker compose up -d` |
| Parar (libera a GPU e a memória) | `docker compose stop` |
| Reiniciar | `docker compose restart` |
| Remover os containers | `docker compose down` (os modelos baixados ficam no cache) |

Um serviço parado com `stop` ou `down` **não volta sozinho** no próximo boot.
Rode `docker compose up -d` para ele voltar a subir automaticamente.

### Depois de mudar o código do CLM

O código fica dentro da imagem, então mudanças em `src/` exigem um novo build:

```bash
cd ~/Workspace/CLM
git pull
./docker/build_jetson.sh      # rápido: só refaz o passo do CLM
docker compose up -d          # recria os containers com a imagem nova
```

### Usar um checkpoint treinado

Coloque o `.pt` em `~/Workspace/CLM/checkpoints/`, que é montada no container
em `/CLM/checkpoints`. Depois aponte o `CLM_CKPT` no `docker/.env` (veja
abaixo) e rode `docker compose up -d`. Só o serviço `clm` é recriado.

---

## Configuração

As variáveis podem vir do ambiente ou de um arquivo `docker/.env` (no mesmo
diretório do compose; ele não vai para o Git). Exemplo:

```bash
# docker/.env
UTIL=0.7
CLM_CKPT=/CLM/checkpoints/robo.pt
```

| Variável | Padrão | Uso |
| --- | --- | --- |
| `UTIL` | `0.7` | `--gpu-memory-utilization` do vLLM. A memória da Jetson é compartilhada com a CPU, então não use 0.9 |
| `MAXLEN` | `2048` | `--max-model-len` do vLLM e truncamento do `clm-serve` |
| `CLM_CKPT` | (cabeça de referência) | Checkpoint treinado, ex.: `/CLM/checkpoints/robo.pt` |
| `POOLING` | `--task embed` | Flag de embedding do vLLM. Use `--runner pooling` com uma base de vLLM ≥ 0.10 |
| `CLM_IMAGE` | `clm-jetson:latest` | Imagem a usar (útil com registry) |
| `HF_TOKEN` | — | Download mais rápido do Hugging Face |
| `CLM_API_KEY` | — | Liga autenticação na API do CLM |

---

## Vários robôs

Construa a imagem uma vez, publique-a num registry e use a mesma tag em todos
os robôs. Os robôs ainda precisam dos passos 1 e 2 da instalação e do
repositório clonado, por causa do compose e de `checkpoints/`.

```bash
# na máquina de build (uma Jetson com o mesmo JetPack):
IMAGE=<registry>/clm-jetson:1.0 ./docker/build_jetson.sh && docker push <registry>/clm-jetson:1.0

# em cada robô, no docker/.env:
CLM_IMAGE=<registry>/clm-jetson:1.0
# e então:
docker compose pull && docker compose up -d
```

Para voltar a uma versão anterior, troque a tag em `CLM_IMAGE` e rode
`docker compose up -d`.

---

## Problemas comuns

| Erro | Causa e solução |
| --- | --- |
| `NVIDIA driver on your system is too old (found version 12060)` | torch/vLLM do PyPI fora do container. Use o container (este guia) |
| `sudo: docker: command not found` | Docker não instalado. Veja o [passo 2](#passo-2-instalar-o-docker-e-o-runtime-nvidia) |
| `permission denied ... /var/run/docker.sock` | Usuário fora do grupo `docker`. Rode `sudo usermod -aG docker $USER` e reconecte |
| `Permission denied` ao instalar dependências do jetson-containers | Instale com `python3 -m pip install --user -r jetson-containers/requirements.txt` |
| `410 Client Error: Gone` para `pypi.jetson-ai-lab.dev` no build | O índice da imagem base foi desativado. O `Dockerfile.jetson` já aponta o pip para o PyPI. Se aparecer, o Dockerfile do robô está desatualizado |
| `libnvrm_gpu.so: file too short` | `import torch` durante o build. As bibliotecas da GPU só existem com o container rodando. Atualize o `Dockerfile.jetson` |
| `can't initialize iptables table 'raw'` | O Docker 28+ usa uma tabela do firewall que o kernel da Jetson não tem. O build e o compose usam rede `host` e não são afetados. Em `docker run` avulso, passe `--network host`. Para reativar a rede `bridge`, veja abaixo |
| `unrecognized arguments: --task` ou `--runner` | `POOLING` não bate com a versão do vLLM da base. vLLM < 0.10 usa `--task embed`, e vLLM ≥ 0.10 usa `--runner pooling` |
| `CUDA out of memory` ou processo morto (OOM) | Reduza `UTIL` (0.6) e feche outros processos que usam GPU. Na Orin 32 GB, os pesos sozinhos ocupam ~16 GB |

Reativar a rede `bridge`, que não foi testada neste robô e só é necessária
para containers sem `--network host`:

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
printf '[Service]\nEnvironment=DOCKER_INSECURE_NO_IPTABLES_RAW=1\n' | \
  sudo tee /etc/systemd/system/docker.service.d/no-iptables-raw.conf
sudo systemctl daemon-reload && sudo systemctl restart docker
```

Isso desliga só uma proteção extra da rede `bridge`: a que impede outra máquina
da rede de acessar diretamente o IP interno de um container.

Para mais desempenho, deixe a placa no modo de potência máxima com
`sudo nvpmodel -m 0` e `sudo jetson_clocks`.
