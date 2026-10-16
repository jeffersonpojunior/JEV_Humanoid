# CLM numa Jetson via container

Guia para rodar o CLM embarcado numa NVIDIA Jetson Orin com JetPack 6: o encoder Qwen3-8B (vLLM, porta 8090) e a API do CLM (`clm-serve`, porta 8700), os dois num container Docker.

Testado numa Jetson Orin 32 GB com **JetPack 6.2 (L4T R36.4.3)**, Ubuntu 22.04, imagem base `dustynv/vllm:0.8.6-r36.4-cu128-24.04` (torch 2.7.0, CUDA 12.8, vLLM 0.8.6).

- [Por que container](#por-que-container)
- [Instalação inicial (uma vez por robô)](#instalação-inicial-uma-vez-por-robô)
- [Uso no dia a dia](#uso-no-dia-a-dia)
- [Configuração](#configuração)
- [Vários robôs](#vários-robôs)
- [Problemas comuns](#problemas-comuns)
- [Desfazer a instalação do Docker](#desfazer-a-instalação-do-docker)

---

## Por que container

**Não use `pip install -e .` direto na Jetson.** O PyPI instala torch e vLLM compilados para CUDA 12.8+/13 e para ARM de servidor (SBSA). Na Orin, o driver vem do JetPack e vai só até CUDA 12.6 (JetPack 6), e o resultado é:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12060)
```

Atualizar o driver não é possível nem necessário. Também não adianta trocar para o índice `cu126` do PyTorch: as wheels aarch64 de lá continuam sendo SBSA. O que resolve é usar torch e vLLM compilados para Jetson, que vêm prontos nas imagens do [jetson-containers](https://github.com/dusty-nv/jetson-containers). A imagem do CLM é construída sobre uma delas:

| Arquivo | Função |
| :-- | :-- |
| `docker/Dockerfile.jetson` | Imagem do CLM sobre a imagem de vLLM para Jetson. Instala o CLM com `--no-deps` e fixa o torch/vLLM da base |
| `docker/build_jetson.sh` | Constrói a imagem. Por padrão usa a base do JetPack 6.2; com `AUTOTAG=1`, escolhe a base compatível com o seu L4T (`autotag vllm`) |
| `docker/docker-compose.jetson.yml` | Define os serviços do encoder (:8090) e do `clm-serve` (:8700); os serviços são iniciados manualmente pelo script `clm.sh` |
| `clm.sh` | Inicia, para, reinicia e mostra o estado/logs dos serviços sob demanda |

---

## Instalação inicial (uma vez por robô)

> **Restrição crítica: não use o `apt` neste robô.**
>
> Não rode `apt update`, `apt upgrade`, `apt full-upgrade`, `apt autoremove`, `apt install` nem `apt --fix-broken install`. Também não use `curl https://get.docker.com | sh` (o script adiciona um repositório, roda `apt-get update` e atualiza pacotes do sistema) nem `apt install nvidia-container` (esse pacote já faz parte do JetPack).
>
> Isso já causou problemas: numa instalação anterior, `apt install -y nvidia-container curl` atualizou 17 metapacotes do JetPack de `6.2+b77` para `6.2.1+b38`, e o script do Docker adicionou um repositório APT e atualizou o `ca-certificates`. Foi preciso reverter à mão (veja [Desfazer a instalação do Docker](#desfazer-a-instalação-do-docker)).
>
> Este guia instala o Docker a partir dos pacotes `.deb` oficiais, com `dpkg`, sem consultar nem atualizar repositórios.

Rode todos os comandos **fora de qualquer venv** (`deactivate` antes).

### Passo 1: conferir o JetPack e o disco

```bash
cat /etc/nv_tegra_release     # esperado: R36 (release), REVISION: 4.x  → JetPack 6.x
df -h / ~                     # precisa de ~30 GB livres

# versão dos metapacotes do JetPack (anote; ex.: 6.2+b77)
dpkg -l | awk '$2 ~ /^nvidia-(jetpack|cuda|cudnn|tensorrt|opencv|vpi|container|cupva|nsight)/ {print $2, $3}'
```

São cerca de 6 GB para a imagem base, 16 GB para o Qwen3-8B e mais alguns GB de folga. Se o disco principal for pequeno e houver um NVMe, mova os dados do Docker para ele depois do [passo 2.6](#26-configurar-o-runtime-nvidia): acrescente `"data-root": "/caminho/no/nvme/docker"` ao `/etc/docker/daemon.json`, **mantendo as chaves que já existem**, e reinicie o Docker. Faça isso antes de construir a imagem.

**Recomendado: travar os metapacotes do JetPack**, para que nenhum `apt` rodado por engano os atualize (o apt passa a recusar o upgrade em vez de fazê-lo em silêncio):

```bash
sudo apt-mark hold nvidia-container nvidia-cuda nvidia-cuda-dev nvidia-cudnn nvidia-cudnn-dev \
  nvidia-cupva nvidia-jetpack nvidia-jetpack-dev nvidia-jetpack-runtime nvidia-nsight-graphics \
  nvidia-nsight-systems nvidia-opencv nvidia-opencv-dev nvidia-tensorrt nvidia-tensorrt-dev \
  nvidia-vpi nvidia-vpi-dev
```

Ajuste a lista se algum pacote não existir no seu robô (o `apt-mark` avisa). Para uma atualização **intencional** do JetPack, rode antes `sudo apt-mark unhold` com os mesmos nomes.

### Passo 2: instalar o Docker por pacotes .deb (sem apt)

O Docker não vem no JetPack. Esta instalação segue o método "Install from a package" da documentação oficial do Docker: baixa os `.deb` e instala com `dpkg`. O apt não é usado, então nenhum pacote do sistema é atualizado.

#### 2.1 Verificar o que já existe

Estes comandos só leem o estado do sistema:

```bash
command -v docker
docker --version
docker compose version
sudo systemctl is-active docker
sudo docker info --format 'Runtimes: {{json .Runtimes}}; Default: {{.DefaultRuntime}}'
dpkg-query -W -f='${binary:Package} ${Version}\n' \
  docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-buildx-plugin \
  nvidia-container-toolkit 2>/dev/null
```

Se `docker`, `docker compose` e o runtime `nvidia` já estiverem disponíveis, **não reinstale nada**: vá para o [Passo 3](#passo-3-trazer-o-repositório-para-o-robô). Se o Docker já existir mas o serviço estiver parado, use `sudo systemctl start docker`. Caso contrário, continue.

#### 2.2 Conferir o toolkit da NVIDIA

O JetPack já traz o `nvidia-container-toolkit` (o runtime `nvidia` e o `nvidia-ctk`). Ele **não** deve ser instalado por este guia:

```bash
dpkg-query -W nvidia-container-toolkit nvidia-container-toolkit-base
command -v nvidia-ctk
```

Se algum deles não aparecer, **pare aqui** e peça ao responsável pela imagem do robô um toolkit validado para a sua versão exata de JetPack/L4T. Não resolva isso com `apt install`.

#### 2.3 Baixar os pacotes

Use os pacotes para **arm64** do Ubuntu 22.04 (`jammy`), na pasta oficial do Docker:

```
https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/arm64/
```

Baixe **um arquivo de cada** (escolha as versões mais recentes de cada pacote, todas do mesmo repositório `jammy`):

| Pacote | Arquivo |
| :-- | :-- |
| containerd | `containerd.io_<versão>_arm64.deb` |
| CLI do Docker | `docker-ce-cli_<versão>_arm64.deb` |
| Docker Engine | `docker-ce_<versão>_arm64.deb` |
| Buildx | `docker-buildx-plugin_<versão>_arm64.deb` |
| Compose | `docker-compose-plugin_<versão>_arm64.deb` |

Não baixe `docker-ce-rootless-extras` nem `docker-model-plugin`: não são necessários.

Referência: estas versões foram instaladas e chegaram a construir a imagem do CLM numa tentativa anterior neste robô: `docker-ce` e `docker-ce-cli` 29.9.0-1, `containerd.io` 2.4.1-2, `docker-buildx-plugin` 0.38.0-1 e `docker-compose-plugin` 5.6.0-1 (todas `~ubuntu.22.04~jammy`).

Baixe em qualquer computador e copie para o robô, ou baixe direto nele se tiver acesso à internet. Coloque os cinco arquivos numa pasta só, sem outras versões misturadas:

```bash
# no robô
mkdir -p ~/docker-debs

# do seu computador, se baixou lá:
scp *.deb <usuario>@<ip-do-robô>:~/docker-debs/

# ou, no próprio robô, um por vez (troque pelo nome exato do arquivo):
cd ~/docker-debs
curl -fLO "https://download.docker.com/linux/ubuntu/dists/jammy/pool/stable/arm64/<arquivo>.deb"
```

Baixe somente de `download.docker.com`, por HTTPS.

#### 2.4 Simular a instalação

```bash
cd ~/docker-debs
ls                                    # exatamente 5 arquivos .deb, um de cada pacote

# dependências declaradas pelo Engine (para conferir à mão, se quiser)
dpkg-deb -f ./docker-ce_*_arm64.deb Depends

# simulação: não grava nada
sudo dpkg --dry-run -i ./containerd.io_*_arm64.deb ./docker-ce-cli_*_arm64.deb \
  ./docker-ce_*_arm64.deb ./docker-buildx-plugin_*_arm64.deb ./docker-compose-plugin_*_arm64.deb
```

Se a simulação terminar sem erros, siga para o 2.5. Se aparecer `dependency problems` ou `depende de ... que não está instalado`, **pare**. Não rode `apt --fix-broken install` nem `apt install -f`: o apt tentaria resolver baixando e atualizando pacotes do sistema. Peça ao responsável pela imagem do robô os pacotes que faltam, já validados para o JetPack.

#### 2.5 Instalar

```bash
cd ~/docker-debs
sudo dpkg -i ./containerd.io_*_arm64.deb ./docker-ce-cli_*_arm64.deb \
  ./docker-ce_*_arm64.deb ./docker-buildx-plugin_*_arm64.deb ./docker-compose-plugin_*_arm64.deb
sudo systemctl is-active docker       # esperado: active
```

O pacote cria o grupo `docker` e habilita o serviço do Docker no boot. Isso **não** faz o CLM iniciar sozinho, porque os containers não têm política de reinício (veja o [Passo 5](#passo-5-configurar-o-início-manual-e-controlar-os-serviços-por-script)). Se você preferir que nem o Docker suba no boot, rode `sudo systemctl disable docker.service docker.socket` e, antes de cada `./clm.sh start`, `sudo systemctl start docker`.

Se o serviço não ficar `active`, veja o motivo com `sudo journalctl -u docker -n 50 --no-pager` e consulte [Problemas comuns](#problemas-comuns).

#### 2.6 Configurar o runtime NVIDIA

Use o `nvidia-ctk` em vez de recriar o `daemon.json` à mão: ele altera só as chaves do runtime e preserva as demais.

```bash
[ -f /etc/docker/daemon.json ] && sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
sudo cat /etc/docker/daemon.json
```

O arquivo deve ficar parecido com:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

Se o `nvidia-ctk` não reconhecer `--set-as-default`, rode o comando sem essa opção e acrescente `"default-runtime": "nvidia"` ao arquivo com `sudo nano /etc/docker/daemon.json`, mantendo o resto, e reinicie o Docker.

#### 2.7 Verificar

```bash
sudo docker info --format 'Runtimes: {{json .Runtimes}}; Default: {{.DefaultRuntime}}'
sudo docker compose version
```

Esperado: `nvidia` na lista de runtimes e `Default: nvidia`.

#### 2.8 Usar o Docker sem `sudo` (opcional)

```bash
sudo usermod -aG docker "$USER"
```

Depois, encerre a sessão e conecte-se de novo (por SSH, reconecte). Confira:

```bash
groups | grep docker
docker info | grep -i runtime
docker compose version
```

A associação ao grupo `docker` concede privilégios equivalentes aos de administrador no host. Se preferir não conceder esse acesso, continue usando `sudo docker` e `sudo docker compose` nos comandos seguintes.

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

A primeira vez baixa a imagem base (5.7 GB). No fim do build aparece uma linha como `torch 2.7.0 | vllm 0.8.6 | clm 0.1.0`.

Se você não entrou no grupo `docker` (passo 2.8), rode o script com `sudo`. Com `sudo`, confira em qual `$HOME` os caches (`~/.cache/huggingface` e `~/.cache/clm`) estão sendo criados: pode ser o do `root`.

O build não importa o torch. Na Jetson, as bibliotecas da GPU só são montadas pelo runtime `nvidia` quando o container roda. Por isso, o teste da GPU vem depois do build:

```bash
docker run --rm --runtime nvidia --network host clm-jetson:latest \
  python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
#   esperado: 2.7.0 12.8 True
```

O `--network host` é necessário: veja [iptables](#problemas-comuns).

### Passo 5: configurar o início manual e controlar os serviços por script

Por padrão, o CLM **não deve iniciar automaticamente no boot**. O Docker pode continuar habilitado no sistema, mas os containers só devem iniciar quando você chamar o script `clm.sh`.

**Antes de iniciar os serviços, edite `docker/docker-compose.jetson.yml`** e remova qualquer linha `restart: unless-stopped` ou `restart: always` de ambos os serviços. Alternativamente, defina explicitamente `restart: "no"` em cada serviço. Isso evita que o Docker reinicie os containers automaticamente após um reboot ou falha.

**Baixe o modelo antes da primeira subida.** Se o vLLM baixar o Qwen3-8B (~16 GB) sozinho, qualquer queda de conexão derruba o encoder, e é preciso começar de novo. Baixe antes, com o mesmo cache que o compose monta. Se cair, rode de novo: os arquivos já completos são aproveitados.

```bash
mkdir -p ~/.cache/huggingface
docker run --rm --runtime nvidia --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_HUB_CACHE=/root/.cache/huggingface/hub \
  -e HF_TOKEN \
  clm-jetson:latest \
  python3 -c "import huggingface_hub.file_download as fd; fd.is_xet_available = lambda: False; from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))"
```

Termina imprimindo o caminho do snapshot (`/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/...`). O `fd.is_xet_available = lambda: False` força o download HTTP comum, mais tolerante a redes instáveis que o protocolo Xet. A imagem traz o `huggingface_hub` 0.30.2, que ignora a variável `HF_HUB_DISABLE_XET` (ela só existe a partir da 0.31), por isso o Xet é desligado direto no código. Durante o download aparece várias vezes o aviso `Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download`. Ele é esperado e confirma que o download está indo por HTTP. Não interrompa. Se o caminho impresso começar com `/data/models/huggingface`, faltaram as variáveis `HF_HOME` e `HF_HUB_CACHE`: a imagem base aponta o cache para essa pasta, que não está montada, e o modelo se perde quando o container termina.

Salve o script `clm.sh` na raiz do repositório, ao lado da pasta `docker/`, e torne-o executável:

```bash
cd ~/Workspace/CLM
chmod +x clm.sh
./clm.sh start
```

O comando inicia os serviços em segundo plano e devolve o terminal. Para acompanhar os logs do encoder:

```bash
./clm.sh logs-encoder
```

O encoder lê o Qwen3-8B de `~/.cache/huggingface`, que fica fora do container e é reaproveitado (se você pulou o download acima, ele baixa os ~16 GB nesta hora).

Se o `start` terminar com `dependency failed to start: container docker-encoder-1 exited (1)`, o `clm` não chegou a subir porque o encoder morreu. A causa está em `./clm.sh logs-encoder` (ou `docker logs --tail 80 docker-encoder-1`); veja [Problemas comuns](#problemas-comuns). O `clm` só inicia quando o encoder responde em `/health`; na primeira vez, ele também baixa a cabeça de referência para `~/.cache/clm`. O encoder leva alguns minutos para carregar o modelo. Para testar:

```bash
curl -s localhost:8090/health && echo encoder ok
curl -s localhost:8700/ >/dev/null && echo clm ok
```

O playground fica disponível em `http://<ip-do-robô>:8700/` enquanto os serviços estiverem ativos.

Teste no playground um estado cuja resposta você já conhece. As embeddings vêm do vLLM 0.8.6 e não da versão usada no treino, então vale conferir se as escolhas fazem sentido.

### Opcional: jetson-containers (só para o `autotag`)

Só é necessário para descobrir a base certa em outra versão do JetPack. Na 6.2, a base padrão do Dockerfile já é a correta, então **pule este passo em robôs com a restrição de apt**: o `install.sh` do jetson-containers pode chamar o apt. Se precisar do `autotag`, rode-o em outra máquina e traga só o nome da base.

```bash
cd ~/Workspace
git clone https://github.com/dusty-nv/jetson-containers
python3 -m pip install --user -r jetson-containers/requirements.txt
bash jetson-containers/install.sh
cd ~/Workspace/CLM && AUTOTAG=1 ./docker/build_jetson.sh
```

Não rode o `install.sh` com o venv do CLM ativo: as dependências vão para o venv e o `autotag` (que usa o `python3` do sistema) falha depois.

---

## Uso no dia a dia

O CLM **não inicia automaticamente com o boot**. Inicie-o apenas quando precisar e pare-o ao terminar. O script `clm.sh` deve ficar na raiz de `~/Workspace/CLM`, ao lado da pasta `docker/`.

```bash
cd ~/Workspace/CLM
./clm.sh start       # inicia encoder e API em segundo plano
./clm.sh status      # mostra o estado dos serviços
./clm.sh logs        # acompanha os logs de ambos (Ctrl+C sai dos logs)
./clm.sh logs-encoder
./clm.sh logs-clm
./clm.sh stop        # para os containers e libera GPU/memória
./clm.sh restart     # reinicia os serviços
./clm.sh down        # para e remove os containers; preserva imagens e caches
```

`stop` mantém os containers para uma próxima inicialização mais simples. `down` remove os containers e a rede criada pelo Compose; os modelos baixados nos diretórios de cache e a imagem Docker são mantidos.

Se o script informar que o usuário não tem acesso ao daemon Docker, use `sudo ./clm.sh start` (ou o subcomando desejado), ou configure as permissões de Docker conforme a política do robô. A associação ao grupo `docker` concede privilégios equivalentes aos de administrador no host.

### Depois de mudar o código do CLM

O código fica dentro da imagem, então mudanças em `src/` exigem um novo build:

```bash
cd ~/Workspace/CLM
git pull
./docker/build_jetson.sh      # rápido: só refaz o passo do CLM
./clm.sh stop
./clm.sh start                # sobe os serviços usando a imagem reconstruída
```

### Usar um checkpoint treinado

Coloque o `.pt` em `~/Workspace/CLM/checkpoints/`, que é montada no container em `/CLM/checkpoints`. Depois aponte o `CLM_CKPT` no `docker/.env` (veja abaixo) e rode `./clm.sh restart` para aplicar a configuração.

---

## Configuração

As variáveis podem vir do ambiente ou de um arquivo `docker/.env` (no mesmo diretório do compose; ele não vai para o Git). Exemplo:

```bash
# docker/.env
UTIL=0.7
CLM_CKPT=/CLM/checkpoints/robo.pt
```

| Variável | Padrão | Uso |
| :-- | :-- | :-- |
| `UTIL` | `0.7` | `--gpu-memory-utilization` do vLLM. A memória da Jetson é compartilhada com a CPU, então não use 0.9 |
| `MAXLEN` | `2048` | `--max-model-len` do vLLM e truncamento do `clm-serve` |
| `CLM_CKPT` | (cabeça de referência) | Checkpoint treinado, ex.: `/CLM/checkpoints/robo.pt` |
| `POOLING` | `--task embed` | Flag de embedding do vLLM. Use `--runner pooling` com uma base de vLLM ≥ 0.10 |
| `CLM_IMAGE` | `clm-jetson:latest` | Imagem a usar (útil com registry) |
| `HF_TOKEN` | — | Download mais rápido do Hugging Face |
| `CLM_API_KEY` | — | Liga autenticação na API do CLM |

---

## Vários robôs

Construa a imagem uma vez, publique-a num registry e use a mesma tag em todos os robôs. Os robôs ainda precisam dos passos 1 e 2 da instalação e do repositório clonado, por causa do Compose, do script `clm.sh` e de `checkpoints/`. Copie os **mesmos cinco `.deb`** do passo 2.3 para cada robô (`scp`), em vez de baixar de novo, para que todos fiquem com a mesma versão do Docker. Mantenha a política de reinício automático desativada no Compose.

```bash
# na máquina de build (uma Jetson com o mesmo JetPack):
IMAGE=<registry>/clm-jetson:1.0 ./docker/build_jetson.sh && docker push <registry>/clm-jetson:1.0

# em cada robô, no docker/.env:
CLM_IMAGE=<registry>/clm-jetson:1.0
# e então:
./clm.sh pull
./clm.sh start
```

Para voltar a uma versão anterior, troque a tag em `CLM_IMAGE`, rode `./clm.sh pull` e depois `./clm.sh start`.

---

## Problemas comuns

| Erro | Causa e solução |
| :-- | :-- |
| `NVIDIA driver on your system is too old (found version 12060)` | torch/vLLM do PyPI fora do container. Use o container (este guia) |
| `docker: command not found` | Docker não instalado. Veja o [passo 2](#passo-2-instalar-o-docker-por-pacotes-deb-sem-apt) (instalação por `.deb`, sem apt) |
| `dpkg: dependency problems prevent configuration of docker-ce` | Falta uma dependência no sistema. Não use `apt install -f`; peça ao responsável pela imagem do robô o pacote que falta, validado para o JetPack |
| `Cannot connect to the Docker daemon` | Serviço parado. `sudo systemctl start docker` e, se falhar, `sudo journalctl -u docker -n 50 --no-pager` |
| `permission denied ... /var/run/docker.sock` | Usuário fora do grupo `docker`. Use `sudo` ou, se a política do robô permitir, `sudo usermod -aG docker $USER` e reconecte |
| `Unknown runtime specified nvidia` ou `could not select device driver "nvidia"` | Runtime NVIDIA não configurado no Docker. Refaça o [passo 2.6](#26-configurar-o-runtime-nvidia) e reinicie o Docker |
| `Permission denied` ao instalar dependências do jetson-containers | Instale com `python3 -m pip install --user -r jetson-containers/requirements.txt` |
| `410 Client Error: Gone` para `pypi.jetson-ai-lab.dev` no build | O índice da imagem base foi desativado. O `Dockerfile.jetson` já aponta o pip para o PyPI. Se aparecer, o Dockerfile do robô está desatualizado |
| `libnvrm_gpu.so: file too short` | `import torch` durante o build. As bibliotecas da GPU só existem com o container rodando. Atualize o `Dockerfile.jetson` |
| `can't initialize iptables table 'raw'` | O Docker 28+ usa uma tabela do firewall que o kernel da Jetson não tem. O build e o compose usam rede `host` e não são afetados. Em `docker run` avulso, passe `--network host`. Para reativar a rede `bridge`, veja abaixo |
| `CAS service error ... IncompleteBody` / `UnexpectedEof` no log do encoder | A conexão caiu durante o download do modelo (protocolo Xet do Hugging Face). Baixe o modelo antes, por HTTP comum, com o comando do [passo 5](#passo-5-configurar-o-início-manual-e-controlar-os-serviços-por-script). `HF_HUB_DISABLE_XET=1` não funciona: o `huggingface_hub` 0.30.2 da imagem ignora essa variável |
| `unrecognized arguments: --task` ou `--runner` | `POOLING` não bate com a versão do vLLM da base. vLLM < 0.10 usa `--task embed`, e vLLM ≥ 0.10 usa `--runner pooling` |
| `CUDA out of memory` ou processo morto (OOM) | Reduza `UTIL` (0.6) e feche outros processos que usam GPU. Na Orin 32 GB, os pesos sozinhos ocupam ~16 GB |

Reativar a rede `bridge`, que não foi testada neste robô e só é necessária para containers sem `--network host`:

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
printf '[Service]\nEnvironment=DOCKER_INSECURE_NO_IPTABLES_RAW=1\n' | \
  sudo tee /etc/systemd/system/docker.service.d/no-iptables-raw.conf
sudo systemctl daemon-reload && sudo systemctl restart docker
```

Isso desliga só uma proteção extra da rede `bridge`: a que impede outra máquina da rede de acessar diretamente o IP interno de um container.

Para mais desempenho, deixe a placa no modo de potência máxima com `sudo nvpmodel -m 0` e `sudo jetson_clocks`.

---

## Desfazer a instalação do Docker

Use esta seção se o Docker foi instalado por engano ou de forma errada (por exemplo, pelo script `get.docker.com`). Ela remove o Docker **sem rodar `apt update` e sem `apt autoremove`**. O `autoremove` age sobre qualquer pacote que o apt considere órfão, não só os do Docker, e já removeu bibliotecas de vídeo e OpenCV de um robô.

**1. Pare e remova os containers do CLM** (se existirem):

```bash
cd ~/Workspace/CLM && ./clm.sh down
```

**2. (Opcional) Guarde a imagem**, se for reutilizá-la com um Docker instalado do jeito certo. Faça isto **antes** de remover o Docker:

```bash
docker save clm-jetson:latest | gzip > ~/clm-jetson.tar.gz     # ~6 GB
```

**3. Remova os pacotes**, simulando primeiro:

```bash
PKGS=$(dpkg -l | awk '/^(ii|rc)/ && $2 ~ /^(docker|containerd)/ {print $2}')
echo $PKGS
sudo apt-get purge -s $PKGS     # só pacotes do Docker/containerd devem aparecer em "Purg"
sudo apt-get purge $PKGS
```

Se a simulação listar qualquer `nvidia-*` ou algo que você não reconheça, **pare** e não confirme. O `purge` também desfaz o mascaramento (`masked`) das unidades `docker.service` e `docker.socket` que uma simples remoção deixa para trás. O apt vai sugerir `autoremove` por causa do `pigz`: ignore a sugestão.

**4. Se o Docker veio do script `get.docker.com`**, remova também o repositório e a chave (sem `apt update`):

```bash
sudo rm -f /etc/apt/sources.list.d/docker.list /etc/apt/keyrings/docker.asc /etc/apt/keyrings/docker.gpg
sudo rm -f /var/lib/apt/lists/download.docker.com_*
```

**5. Remova os restos:**

```bash
sudo rm -rf /var/lib/docker /var/lib/containerd /etc/docker /etc/systemd/system/docker.service.d
sudo gpasswd -d "$USER" docker
sudo groupdel docker
sudo ip link delete docker0
```

O `rm -rf /var/lib/docker` apaga imagens e cache de build (dezenas de GB). Erros de `gpasswd`, `groupdel` ou `ip link` que digam que o grupo ou a interface não existem podem ser ignorados.

**6. Verifique:**

```bash
dpkg -l | grep -E 'docker|containerd'        # esperado: nenhuma saída
command -v docker                            # esperado: nenhuma saída
systemctl is-enabled docker 2>&1             # esperado: erro "No such file or directory" / not-found
ls -l /etc/systemd/system/ | grep -i docker  # esperado: nenhuma saída
ip link show docker0 2>&1                    # esperado: "does not exist"
```

As regras de firewall (as chains `DOCKER*` do iptables) só desaparecem de vez com um reboot, que pode ser feito quando for conveniente para o robô.
