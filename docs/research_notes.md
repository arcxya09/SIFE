# KrF 激光直接驱动惯性约束聚变：物理调研笔记

本笔记是 SIFE（Simulation of Inertial Fusion Energy）程序的物理基础。内容依次为：完整物理过程链，每个环节的主流模型、SIFE 采用的模型及其适用范围，主流程序的对照，以及参考文献。

---

## 1. KrF 直接驱动的特点

| 参数 | KrF | Nd:glass 三倍频 (3ω) | ArF |
|---|---|---|---|
| 波长 λ | **248 nm** | 351 nm | 193 nm |
| 临界密度 n_c = 1.115×10²¹/λ²[μm] cm⁻³ | **1.81×10²² cm⁻³** | 9.0×10²¹ | 3.0×10²² |
| 带宽 | 宽（THz 量级），可用 ISI 光学平滑 | 窄（需 SSD 等技术） | 宽 |

KrF 驱动的优势来自三个方面：

- **吸收更好**：逆轫致吸收系数 κ_ib ∝ n_e²/(n_c T_e^{3/2}) ∝ λ⁻²，在相同光强下吸收区密度更高，吸收率更高。在临界密度附近，烧蚀压力按 P_a ∝ (I/λ)^{2/3} 定标，波长更短时更高。
- **激光-等离子体不稳定性 (LPI) 阈值更高**：受激拉曼散射、双等离子体衰变和交叉光束能量转移的阈值都更高；宽带也能抑制 LPI。热电子预热因此更弱，可以用更高的光强和更低的熵增（更低绝热因子）。
- **ISI 平滑与"变焦"(zooming)**：内爆中可以缩小焦斑，使其跟随临界面，减少折射造成的能量浪费，提高耦合效率。

NRL（美国海军研究实验室）的 Nike 和 Electra 装置、Bodner、Obenschain、Schmitt 等人的工作，给出了 0.5–2.5 MJ KrF 驱动的高增益设计：采用 DT 浸润 CH 泡沫烧蚀层加 DT 冰，部分方案使用冲击点火 (shock ignition)，一维增益可达 100–250。

## 2. 物理过程链与 SIFE 的建模

```
激光 ──(逆轫致吸收, 射线追踪)──► 冕区电子 ──(限流 Spitzer 热传导)──► 烧蚀面
   烧蚀压力 ──► 冲击波序列压缩燃料 (绝热线) ──► 壳层加速 (内爆速度 3–4×10⁷ cm/s)
   ──► 滞止: 热斑 (PdV 做功, 电子-离子弛豫, 轫致辐射损失, 热传导损失)
   ──► DT 聚变 (Bosch-Hale 反应率) ──► α 粒子沉积 (自加热) ──► 点火/燃烧波 ──► 能量释放
```

### 2.1 流体力学
- **模型**：一维拉格朗日交错网格（von Neumann–Richtmyer），预估-校正时间推进，人工粘性 q = ρ(C₂Δu² + C₁c_sΔu)。动量方程使用相容面积 dV/dr，保证能量守恒。
- **双温**：电子与离子各自有内能方程。冲击波粘性加热计入离子（物理上冲击波加热离子），电子通过 e-i 弛豫获得能量。辐射能量随流体压缩做 P_r dV 功（P_r = E_r/3）。
- **验证**：Sedov 点爆炸、Noh 球形内爆。

### 2.2 物态方程 (EOS)
- **主流**：SESAME 表、QEOS（Thomas-Fermi 电子 + Cowan 离子 + Barnes 键合修正）、FPEOS（第一性原理）。
- **SIFE v0.1**：采用解析物态：
  - 电子是完全电离的理想费米气体，使用有效温度 T_eff = √(T_e² + T_F²)，T_F = 0.4 E_F。它在经典极限和完全简并极限下都精确，能给出简并压 P_deg = 2.17 ρ^{5/3} Mbar（DT），决定燃料绝热因子 α = P/P_deg。
  - 离子为理想气体。
  - 固体（DT 冰、CH）加入只依赖密度的冷压修正：在 ρ₀ 处总压为零、体模量为 B₀，高压缩时回到费米压；ρ < ρ₀ 时抵消简并压，并带有界张力（最大 0.2 B₀，ρ < 0.6ρ₀ 时消失），防止冷冰表面向低压气体虚假"升华"。
- **局限**：没有部分电离（Z* 为常数）、电离能和分子解离。低温段（< 10 eV）的冲击 Hugoniot 不精确，影响第一冲击的定时与熵。下一步可接入表格 EOS 接口，例如 FEOS/SESAME 风格的表。

### 2.3 激光能量沉积
- **主流**：三维射线追踪加逆轫致吸收（LILAC、DRACO、MULTI、HYDRA），CBET（交叉光束能量转移）模型；3ω 直接驱动中，CBET 可损失 10–30% 的耦合效率。
- **SIFE**：一维球对称射线追踪。
  - 光线守恒量 n r sin ψ = b（Bouguer 定律），n² = 1 − n_e/n_c。
  - 光学厚度 τ = ∫ ν_ei(n_e) (n_e/n_c) r dr / (c √(r²n² − b²))。n_e 在相邻区心之间线性插值，每个网格内解析积分，包括转折点处的可积平方根奇点。
  - 多束重叠照明用一组按焦斑强度分布（超高斯）取冲击参数的光线等效，支持 KrF 变焦。
  - 吸收系数 κ_ib = ν_ei X/(c√(1−X))，ν_ei 为 Spitzer 碰撞频率，Z 取 ⟨Z²⟩/⟨Z⟩。
- **验证**：线性密度剖面的吸收率与解析解（Kruer）以及数值积分比较，0–60° 入射角相对误差 < 0.2%。
- **未含**：Langdon 效应（高强度下吸收降低）、CBET（KrF 宽带下较弱）、共振吸收、热电子。

### 2.4 电子 / 离子热传导
- **主流**：Spitzer-Härm 加通量限制因子 f ≈ 0.06；更先进的有非局域模型（SNB、Schurtz-Nicolaï），以及考虑简并的 Lee-More 电导率。
- **SIFE**：
  - 电导率为 κ_e = γ(Z) n_e T τ_e/m_e，γ(Z) = 13.58 (Z+0.24)/(Z+4.2)（Z=1 时 3.16，Lorentz 极限 128/3π）。
  - 调和型通量限制：q = q_SH/(1 + |q_SH|/(f q_fs))。
  - 冷密燃料中用 T_eff 代替 T，模拟简并对碰撞的抑制。
  - 离子热传导采用 Braginskii 3.9 n_i T_i τ_i/m_i。
- **验证**：Zel'dovich–Raizer 非线性热波点源自相似解，L1 误差 < 0.6%。

### 2.5 电子-离子能量交换
Q_ei = 3(m_e/m_i) n_e (T_e − T_i)/τ_e（Braginskii），与热传导、辐射在同一个隐式步中耦合求解。

### 2.6 辐射输运
- **主流**：多群扩散或 S_N / IMC 输运，配合非 LTE 或 LTE 不透明度表（PROPACEOS 等）。
- **SIFE**：灰体限流扩散（Wilson 和式限流），三温 (T_e, T_i, T_r) 耦合。
  - 不透明度取自由-自由（Kramers 轫致）Planck 平均（由 NRL 轫致功率 1.69×10⁻³² n_e √T_e Σ Z² n_i W/cm³ 校准）。
  - Rosseland 平均 = Planck/30.26（数值积分得到），并加入 Thomson 散射。
  - 外边界为 Marshak 真空条件。
  - 吸收采用双温谱修正：G(T_e,T_r) = (T_e/T_r)³[ψ(1+T_r/T_e)+γ_E]（Planck 谱在 T_r 下的自由-自由吸收）。热斑或冕区的硬 X 射线在冷密燃料中不会像单温灰体近似那样被过度吸收；T_e = T_r 时与平衡一致。
  - 这一模型正确描述 DT 热斑的轫致辐射损失，这对点火判据至关重要。
- **局限**：CH 的束缚-自由和线辐射没有包含（只预留了倍乘因子 `opac_mult`），烧蚀层辐射预热会被低估。

### 2.7 核反应与燃烧
- **反应率**：Bosch & Hale (1992) DT 参数化公式，已与原文表格逐点核对（1–100 keV，4 位有效数字一致）。
- **燃料消耗**：dy/dt = −(ρ⟨σv⟩/2) y²，在每步内解析积分。
- **能量分配**：17.6 MeV = 3.5 MeV α + 14.1 MeV 中子。中子假设逃逸，只做统计；高 ρR 时中子沉积约几个百分点，暂未计入。
- **验证**：等压热斑点火阈值 (ρR)_h 在 5、8、12 keV 时分别为 0.30、0.15、≤0.10 g/cm²。预加热 DT 球的燃耗与经典公式 Φ = ρR/(ρR + H_B)（H_B ≈ 6–7 g/cm²）比较。ρR ≤ 3 时一致，ρR = 10 时低约 12%，属于该经验公式自身的精度范围。

### 2.8 α 粒子输运
- **主流**：多群扩散（Corman）、直线/弯曲轨迹的蒙特卡罗或确定性射线方法。阻止本领可选 Spitzer/Fraley 经典公式、Li-Petrasso 或 BPS 模型。
- **SIFE**：多角度（Gauss-Legendre μ）直线射线慢化输运。阻止本领采用 Trubnikov/NRL 公式：dε/dt = −2[(m_α/m_f)ψ(x) − ψ′(x)]ν₀ ε，同时给出电子和离子的分配，覆盖 v_α ≪ v_te 与 v_α ≫ v_ti 两个极限。
- **验证**：离子获能比例与 Fraley 公式 T/(32+T) 一致（偏差 < 25%，高温段 < 5%）。射程比 Atzeni & Meyer-ter-Vehn 的拟合公式长 1.5–1.8 倍，差异主要来自库仑对数（热斑条件下 SIFE 用 NRL 形式，lnΛ ≈ 4–5）。这意味着 SIFE 对 α 自加热的估计偏保守，详见第 4 节"待改进"。

## 3. 关键设计参数与定标律（用于结果判读）

- **烧蚀压力**（直接驱动）：P_a ≈ 40–60 (I₁₅/λ_μm)^{2/3} Mbar。
- **燃料绝热因子** α = P/P_deg，P_deg = 2.17 ρ^{5/3} Mbar。高增益设计取 α ≈ 1.5–3。
- **内爆速度**：点火需要 v_imp ≈ 3–4×10⁷ cm/s。点火能量 E_ign ∝ α^{1.8} v^{-6} P_a^{-0.8}（Herrmann-Tabak-Lindl）。
- **热斑点火判据**：(ρR)_hs ≳ 0.3 g/cm²，T_hs ≳ 5–10 keV（Lawson 型 ρR·T 判据）。
- **增益**：G = η_abs · η_hydro · (E_fus/E_shell)。一维 KrF 设计在 1 MJ 级可达 G ≈ 100。
- **流体效率**：直接驱动 η_hydro ≈ 5–12%。

## 4. 现有主流程序对照

| 程序 | 单位 | 维度 | 特点 |
|---|---|---|---|
| LILAC | LLE | 1D | 直接驱动标准设计程序，3D 射线追踪 + CBET，非局域热传导 |
| DRACO | LLE | 2D/3D | ALE，多群辐射扩散 |
| FAST | NRL | 2D/3D | 欧拉 FCT，NRL 的 KrF 设计主力 |
| HYDRA / LASNEX | LLNL | 2D/3D | 多物理，间接驱动 |
| MULTI-IFE | 西班牙 Ramis 等 | 1D | 开放源码，多群辐射 + 射线追踪 + α 输运 |
| HELIOS | Prism | 1D | 商业，教学/实验分析 |
| HYADES | Cascade | 1D | 商业 |
| 国内 | IAPCM 等 | 1D/2D | LARED、MULTI-2D 改编、RDMG 等 |

SIFE 的架构参考 MULTI-IFE（Ramis & Meyer-ter-Vehn 2016）与 LILAC 的一维物理组合。

## 5. 已知局限与路线图

1. **EOS / 不透明度表格化**：接入 SESAME/FEOS 格式表与多群不透明度表，并加入部分电离 Z*(ρ,T)（More 的 TF 拟合）。
2. **非局域电子热传导**（SNB 模型），以及热电子预热。
3. **多群辐射扩散**：CH 冕区辐射预热对绝热线影响明显。
4. **α 阻止本领**：可选 Li-Petrasso/BPS，并提供 lnΛ 选项；**中子沉积**。
5. **二维**：轴对称 ALE 拉格朗日，用于 Rayleigh-Taylor 不稳定性、靶面粗糙度和激光非均匀性。当前的模块化接口（EOS、输运系数、射线追踪、燃烧都是按区函数）可直接复用。
6. **冲击点火**脉冲（末端尖峰）的自动定时优化。

## 6. 参考文献

1. S. Atzeni, J. Meyer-ter-Vehn, *The Physics of Inertial Fusion*, Oxford (2004)：燃烧效率、点火判据、α 射程。
2. H.-S. Bosch, G.M. Hale, Nucl. Fusion **32**, 611 (1992)：DT 反应率参数化。
3. W.L. Kruer, *The Physics of Laser Plasma Interactions* (1988)：逆轫致吸收。
4. L. Spitzer, R. Härm, Phys. Rev. **89**, 977 (1953)；E.M. Epperlein, M.G. Haines, Phys. Fluids **29**, 1029 (1986)：热传导。
5. S.I. Braginskii, Rev. Plasma Phys. **1** (1965)：输运系数与 e-i 交换。
6. NRL Plasma Formulary（库仑对数、轫致辐射功率、慢化率 ν_ε）。
7. G.S. Fraley, E.J. Linnebur, R.J. Mason, R.L. Morse, Phys. Fluids **17**, 474 (1974)：α 能量分配与燃烧。
8. Ya.B. Zel'dovich, Yu.P. Raizer, *Physics of Shock Waves and High-Temperature Hydrodynamic Phenomena*：非线性热波自相似解。
9. W.F. Noh, J. Comput. Phys. **72**, 78 (1987)：Noh 问题。
10. S.E. Bodner et al., Phys. Plasmas **7**, 2298 (2000)：高增益 KrF 直接驱动靶设计。
11. A.J. Schmitt et al., Phys. Plasmas **17**, 042701 (2010)：KrF 冲击点火靶设计。
12. S.P. Obenschain et al., "Progress in the science and technology of direct drive laser fusion with the KrF laser"（FPA 2010 报告）：https://fire.pppl.gov/fpa10_KrF_obenschain.pdf
13. R. Ramis, J. Meyer-ter-Vehn, Comput. Phys. Commun. **203**, 226 (2016)：MULTI-IFE。
14. J.J. MacFarlane et al., JQSRT **99**, 381 (2006)：HELIOS。
15. R.S. Craxton et al., Phys. Plasmas **22**, 110501 (2015)：直接驱动综述。
16. KrF 设计会议摘要：Schmitt et al., APS DPP 2008 "Target designs for ICF using ~1 MJ of direct KrF laser light"：https://archive.aps.org/dpp/2008/to5/14
