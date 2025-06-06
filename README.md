# 🦾 dl_grasping_loco_manipulation_2025

Este repositório contém o desenvolvimento de simulações robóticas utilizando a plataforma **Genesis**, uma plataforma de simulação física voltada para aplicações em **Robótica, Inteligência Artificial Incorporada e Inteligência Física**.

🔗 **Mais informações sobre o Genesis:**
    📖 Documentação: [Genesis Docs](https://genesis-world.readthedocs.io/en/latest/)
    💻 Repositório oficial: [Genesis GitHub](https://github.com/Genesis-Embodied-AI/Genesis)

## 📌 Objetivo do Projeto

Este trabalho está sendo desenvolvido no âmbito da pesquisa do *Mobile Robotics Group* do Laboratório de Robótica da Escola de Engenharia de São Carlos (EESC), focada em *Otimização de Grasping em Robôs Quadrúpedes com braço*. O objetivo é aprimorar a manipulação locomotiva utilizando *Deep Learning*, com foco principal em implementações high-level utilizando o braço robótico acoplado no **Spot da Boston Dynamics**.

A pesquisa se inspirou no estudo abaixo:
    📄 ["Simultaneous Multi-View Object Recognition and Grasping in Open-Ended Domains"](https://link.springer.com/article/10.1007/s10846-024-02092-5)

## 🏗️ Estrutura do Repositório

```text
/
├── urdf/                                  # Arquivos de descrição dos robôs (URDF, meshes, STEP, etc.)
│
├── scripts/                               # Scripts para simulações do Spot e análises
│   ├── spot_arm/                          # Simulações de grasping com o braço do Spot
│   ├── spot_gripper/                      # Simulações de grasping utilizando apenas o gripper do Spot
│   │   ├── normal_grasp/                  # Exploração do alinhamento de normais com o gripper em simulação
│   │   │   └── d2nt/                      # Validação do alinhamento de normais com mapas gerados pelo D2NT
│
├── dataset/                               # Scripts e notebooks para extração de amostras no Genesis
│   ├── bottles/                           # Modelos 3D das garrafas usadas nas simulações de grasping
│   ├── final/                             # Versão final dos notebooks otimizados para extração do dataset
│   └── test/                              # Prototipações e testes de extração de dataset
│
├── grasp_selection/                       # Treinamento de modelos de redes neurais para seleção de pixels de grasping
│   ├── model/                             # Modelos treinados salvos
│   └── data/                              # Dados de teste utilizados nas primeiras iterações de treinamento
│
├── spot_deploy/                           # Scripts de deploy no Spot real, usando a SDK da Boston Dynamics
│   ├── evaluation/                        # Scripts da pipeline final de deploy: integração entre D2NT, YOLO e Grasp_NN
│   ├── images/                            # Dados extraídos da câmera do gripper do Spot
│   └── tunning/                           # Scripts para depuração e ajuste fino de parâmetros do deploy
│
└── README.md                              # Este arquivo
```

## 🚀 Configuração do Ambiente

Para rodar as simulações, siga os passos:

1. **Criar um ambiente virtual**:

   ```bash
   conda create -n genesis_env python=3.12
   conda activate genesis_env
   ```

   ou utilizando `pyenv`:

   ```bash
   pyenv virtualenv 3.12 genesis_env
   pyenv activate genesis_env
   ```

2. **Instalar o Genesis**:
   📖 Guia de instalação: [Genesis Installation](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html)

   ```bash
   pip install genesis-world
   ```

3. **Configuração para GPU (Opcional)**:
   Caso sua máquina possua uma GPU NVIDIA, recomenda-se [configurar os drivers](https://developer.nvidia.com/cuda-downloads) para melhor desempenho.

## 🤖 Desenvolvimento e Testes no Robô Real

Além das simulações, foram realizados testes no robô Spot utilizando a **SDK e API da Boston Dynamics**:

* 🔗 [Spot SDK - Exemplos em Python](https://github.com/boston-dynamics/spot-sdk/tree/7569b7998d486109f80de31dd5f86470016bb141/python/examples)
* 🔗 [Spot API - Protobuf](https://github.com/boston-dynamics/spot-sdk/tree/7569b7998d486109f80de31dd5f86470016bb141/protos/bosdyn/api)

## 📸 Geração de Mapas de Normais

Para melhorar o grasping, foram utilizados mapas de normais a partir de câmeras de profundidade. No **Genesis**, há ferramentas nativas para isso. No mundo real, utilizamos:

* 🔗 [D2NT - Depth to Normal Transformation](https://mias.group/D2NT/)

## 📖 Documentação e APIs do Genesis

* 📘️ [User Guide](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)
* 🔧 [API Reference](https://genesis-world.readthedocs.io/en/latest/api_reference/index.html)

## 🤝 Contribuições

Este projeto está em desenvolvimento contínuo! Se deseja contribuir ou tem sugestões, abra uma **issue** ou envie um **pull request**. 💡🚀
