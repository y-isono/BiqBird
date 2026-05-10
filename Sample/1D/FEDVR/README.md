# 1-D H/He atom — FEDVR (Finite-Element Discrete Variable Representation) implementation

**有限要素離散変数表現 (FEDVR)** を空間離散化に用い、虚時間 RK4 緩和で
1次元ソフトクーロン H/He 原子の電子基底状態を求める実装。

物理ハミルトニアンは親ディレクトリ [`../README.md`](../README.md)、
有限差分版との比較は [`../FD/`](../FD) を参照。

---

## 1. 物理ハミルトニアン（規約）

すべて原子単位、ソフトクーロン（ $a = 1$ ）。

$$ \hat{h}_Z(x) = -\frac{1}{2} \frac{d^2}{dx^2} - \frac{Z}{\sqrt{x^2+1}} $$

$$ w(x_1,x_2) = \frac{1}{\sqrt{(x_1-x_2)^2 + 1}} $$

- **H 原子** （ $Z = 1$ , 1電子）： $\hat{H}_{\rm H} = \hat{h}_1$
- **He 原子** （ $Z = 2$ , 閉殻 2電子, 単一空間軌道 $\psi$ ）：

$$ E[\psi] = 2 \langle \psi | \hat{h}_2 | \psi \rangle + \iint |\psi(x_1)|^2 w(x_1,x_2) |\psi(x_2)|^2 dx_1 dx_2 $$

虚時間 Schrödinger 方程式

$$ \partial_\tau \psi = -\hat{F}[\psi]\, \psi $$

を RK4 で時間発展（毎ステップ正規化）して基底状態に緩和する。He の場合

$$ \hat{F}[\psi] = \hat{h}_2 + \hat{W}_\rho $$

で、 $\hat{W}_\rho$ は密度 $\rho = \lvert \psi \rvert^2$ から作る Hartree ポテンシャル（自己相互作用込み・閉殻 2 電子なので係数 1）。

---

## 2. FEDVR の構成

### 2.1 要素分割

区間 $[-L, L]$ を $N_e$ 個の有限要素に均等分割する：

$$ -L = X_0 < X_1 < \cdots < X_{N_e} = L $$

$$ \Delta_i = X_{i+1} - X_i = \frac{2L}{N_e} $$

各要素 $i$ の区間

$$ \mathcal{E}_i = [X_i, X_{i+1}] $$

の内部に **Gauss–Lobatto–Legendre (GLL) 求積点** （次数 $n$ 、 $n + 1$ 点、両端点を含む）と重み $w^{(i)}_k$ を取る：

$$ \xi^{(i)}_0 = X_i, \quad \xi^{(i)}_1, \ldots, \xi^{(i)}_{n-1}, \quad \xi^{(i)}_n = X_{i+1} $$

実装上は標準区間 $[-1,1]$ の GLL 点 $\hat{\xi}_k$ と重み $\hat{w}_k$
（ただし $\hat{\xi}_k$ は $P'_n(\xi) = 0$ の根 + 端点 $\pm 1$ 、 $\hat{w}_k = 2/[n(n+1) P_n(\hat{\xi}_k)^2]$ ）
をニュートン法で求めて、各要素にアフィン変換でマップ：

$$ \xi^{(i)}_k = \frac{X_i + X_{i+1}}{2} + \frac{\Delta_i}{2}\hat\xi_k $$

$$ w^{(i)}_k = \frac{\Delta_i}{2}\hat w_k $$

### 2.2 局所基底（Lobatto 補間関数）

要素 $\mathcal{E}_i$ 内で点 $\xi^{(i)}_k$ に紐付いた **Lagrange 補間多項式** （次数 $n$ ）：

$$ f^{(i)}_k(x) = \prod_{l \neq k} \frac{x - \xi^{(i)}_l}{\xi^{(i)}_k - \xi^{(i)}_l}, \quad x \in \mathcal E_i $$

要素外では 0。この関数族は **DVR 性** を満たす：

$$ f^{(i)}_k(\xi^{(i)}_l) = \delta_{kl} $$

### 2.3 ブリッジ関数（要素境界の連続化）

要素 $i-1$ の右端点 $\xi^{(i-1)}_n = X_i$ と、要素 $i$ の左端点 $\xi^{(i)}_0 = X_i$ は同一座標。
両側のローカル基底を線形結合し、規格化したものを **ブリッジ関数** $\chi_i$ とする（ $i = 1, \ldots, N_e - 1$ ）：

$$ \chi_i(x) = \frac{f^{(i-1)}_n(x) + f^{(i)}_0(x)}{\sqrt{w^{(i-1)}_n + w^{(i)}_0}} $$

これにより波動関数は要素境界で **C⁰ 連続**（一階導関数は不連続でも OK）になる。

### 2.4 大域 DVR 基底とインデクシング

最終的に独立な DVR 基底は次の組み合わせ：

| 種類 | 個数 | 備考 |
|---|---|---|
| 要素内部点 | $N_e (n - 1)$ | 各要素 $i$ の $k = 1, \dots, n-1$ |
| ブリッジ点 | $N_e - 1$ | 要素境界（左端 $X_0$ 、右端 $X_{N_e}$ は除外＝Dirichlet 0境界条件） |
| **合計** | $N = N_e (n - 1) + (N_e - 1) = N_e n - 1$ |

各大域インデックス $\alpha \in \\{0, \dots, N-1\\}$ に対し、基底関数 $\phi_\alpha$ は次のように規格化される。
$\alpha$ が $i$ 番要素の内部点（ローカル番号 $k \in \\{1, \dots, n-1\\}$ ）に対応する場合は

$$ \phi_\alpha(x) = f^{(i)}_k(x) / \sqrt{w^{(i)}_k} $$

$\alpha$ がブリッジ点（要素境界 $X_i$ , $i = 1, \dots, N_e - 1$ ）に対応する場合は

$$ \phi_\alpha = \chi_i $$

両端点 $X_0, X_{N_e}$ は対応基底を採用しない（境界 Dirichlet 0）。

DVR 性により：

$$ \phi_\alpha(x_\beta) = \frac{\delta_{\alpha\beta}}{\sqrt{w_\alpha}} $$

$$ \langle \phi_\alpha | \phi_\beta \rangle \approx \delta_{\alpha\beta} \quad \text{(GLL 求積)} $$

ここで $w_\alpha$ は「DVR 重み」と呼ばれる量で、要素内部点では

$$ w_\alpha = w^{(i)}_k $$

ブリッジ点 $X_i$ では

$$ w_\alpha = w^{(i-1)}_n + w^{(i)}_0 $$

である。波動関数 $\psi(x) = \sum_\alpha c_\alpha \phi_\alpha(x)$ の係数は
**規格化点値** $c_\alpha = \sqrt{w_\alpha}\, \psi(x_\alpha)$ となり、 $\sum_\alpha \lvert c_\alpha \rvert^2 = 1$ で L²規格化される。

---

## 3. 行列要素

### 3.1 局所ポテンシャル（対角）

任意の局所演算子 $V(x)$ は DVR 基底で対角：

$$ V_{\alpha\beta} = \langle \phi_\alpha | V | \phi_\beta \rangle \approx V(x_\alpha) \delta_{\alpha\beta} $$

これが DVR 最大の利点。外部ポテンシャル $V_{\rm ne}(x) = -Z / \sqrt{x^2 + 1}$ も Hartree ポテンシャルも対角ベクトルとして扱う。

### 3.2 運動エネルギー行列

$$ T_{\alpha\beta} = \frac{1}{2} \int \phi'_\alpha(x) \phi'_\beta(x) dx = \frac{1}{2} \sum_i \int_{\mathcal E_i} \phi'_\alpha \phi'_\beta dx $$

要素内では Lagrange 多項式の導関数を GLL 点で求積する：

$$ \int_{\mathcal E_i} g'_k(x) g'_l(x) dx \approx \sum_{m=0}^{n} w^{(i)}_m D^{(i)}_{mk} D^{(i)}_{ml} $$

$$ D^{(i)}_{mk} = f'^{(i)}_k(\xi^{(i)}_m) $$

要素 $i$ 内部の Lagrange 微分行列 $D^{(i)}$ （標準 GLL の場合 $D^{(i)} = (2 / \Delta_i)\, \hat{D}$ ）の閉形式：

$$
\hat D_{mk} = \begin{cases}
\dfrac{P_n(\hat\xi_m)}{P_n(\hat\xi_k)}\dfrac{1}{\hat\xi_m - \hat\xi_k} & (m \neq k) \\
-\dfrac{n(n+1)}{4} & (m = k = 0) \\
+\dfrac{n(n+1)}{4} & (m = k = n) \\
0 & (\text{otherwise})
\end{cases}
$$

これを使って **要素ローカル運動行列** ($n+1$ 次元実対称)：

$$ T^{(i)}_{kl} = \frac{1}{2} \sum_{m=0}^{n} w^{(i)}_m D^{(i)}_{mk} D^{(i)}_{ml} $$

を作り、規格化（ $\phi$ は $\sqrt{w}$ で割る）を反映してグローバル行列に組み込む。グローバル成分（要素内点 $\alpha = (i, k)$ , $\beta = (i, l)$ が同要素のとき）は

$$ T_{\alpha\beta} = \frac{T^{(i)}_{kl}}{\sqrt{w_\alpha}\sqrt{w_\beta}} $$

ブリッジ点が関わる場合は隣接 2 要素の寄与を、ブリッジ規格化 $\sqrt{w^{(i-1)}_n + w^{(i)}_0}$ で割って足し合わせる。

最終的に $T$ は **対称・バンド幅 $\sim 2n + 1$ の疎行列** 。Eigen の `SparseMatrix<double>` で構築する。

### 3.3 Hartree 項（DVR 近似）

$$ W_{\rho}(x_\alpha) = \int w(x_\alpha, x') \rho(x') dx' \approx \sum_\beta w(x_\alpha, x_\beta) w_\beta \rho_\beta $$

ここで密度 $\rho_\beta = \lvert \psi(x_\beta) \rvert^2 = \lvert c_\beta \rvert^2 / w_\beta$ なので、 **点値演算** で

$$ W_\alpha = \sum_\beta \frac{\lvert c_\beta \rvert^2}{\sqrt{(x_\alpha - x_\beta)^2 + 1}} $$

（重みが綺麗にキャンセルする）。Hartree ポテンシャル $W_\alpha$ は対角で軌道に作用：

$$ (\hat W \psi)_\alpha = W_\alpha c_\alpha $$

---

## 4. 内積・規格化

DVR 係数表現では単純に

$$ \langle \psi_1 | \psi_2 \rangle \approx \sum_\alpha c^{(1)*}_\alpha c^{(2)}_\alpha $$

$$ \lVert \psi \rVert^2 = \sum_\alpha \lvert c_\alpha \rvert^2 $$

毎ステップ後に $c_\alpha \leftarrow c_\alpha / \lVert \psi \rVert$ で正規化する。

---

## 5. 虚時間プロパゲーション (RK4)

$$ \partial_\tau \mathbf c = -\hat F[\mathbf c] \mathbf c $$

$$ \hat F = T + \mathrm{diag}(V_{\rm ne}) + \mathrm{diag}(\hat W_{\rho}) \quad (\text{He のみ}) $$

を 4 段 RK4 で進めて毎ステップ正規化（`FD/He.hpp` の `rk4_gs` と同じパターン）。

---

## 6. ファイル一覧（実装後）

| ファイル | 内容 |
|---|---|
| `fedvr_basis.hpp` | `FEDVRGrid` クラス：GLL 点・重み・運動エネルギー行列 $T$ （疎）と座標 $x_\alpha$ 、 $w_\alpha$ 。第二量子化用の積分テンソル `build_h_pq` 、 `build_V_pq` も提供 |
| `H_fedvr.hpp` / `H_fedvr_GS.cpp` | 1電子（H 原子）。直接対角化との一致で検証 |
| `He_fedvr.hpp` / `He_fedvr_GS.cpp` | 閉殻 He、Hartree + 虚時間 RK4（第一量子化） |
| `hf_fedvr.hpp` / `He_fedvr_HF.cpp` | 閉殻 RHF（第二量子化）。虚時間 RK4 と Roothaan-Hall SCF の両方を提供（§11） |
| `Be_fedvr_HF.cpp` | 閉殻 Be 原子（ $Z = 4$ , $N_{\rm occ} = 2$ ）の RHF。同じ `hf_fedvr.hpp` 機構を $N_{\rm occ} > 1$ に拡張（§12） |

## 7. ビルド・実行例

`Eigen` は `brew install eigen` でインストール（macOS, Apple Silicon 想定で `/opt/homebrew/include/eigen3`）。

```bash
mkdir -p out
EIGEN=/opt/homebrew/include/eigen3
g++ -std=c++17 -O2 -I"$EIGEN" H_fedvr_GS.cpp  -o out/h_fedvr_gs  && ./out/h_fedvr_gs
g++ -std=c++17 -O2 -I"$EIGEN" He_fedvr_GS.cpp -o out/he_fedvr_gs && ./out/he_fedvr_gs
g++ -std=c++17 -O2 -I"$EIGEN" He_fedvr_HF.cpp -o out/he_fedvr_hf && ./out/he_fedvr_hf
g++ -std=c++17 -O2 -I"$EIGEN" Be_fedvr_HF.cpp -o out/be_fedvr_hf && ./out/be_fedvr_hf
```

## 8. 既定パラメータ

| | $L$ | $N_e$ | $n$ | DOF $N$ | $d\tau$ | max_itr |
|---|---|---|---|---|---|---|
| H  | 20 | 10 | 8 | 79 | 0.05 | 10000 |
| He | 20 | 10 | 8 | 79 | 0.20 | 1000 |

DOF $N = N_e n - 1$（両端 Dirichlet）

## 9. Reference values & verification

ソフトクーロン規約 $a = 1$ における 1D He の基底状態エネルギー：

| 計算 | $E_{\rm tot}$ [a.u.] | 備考 |
|---|---|---|
| **Exact (FCI)** | **-2.23825730** | [Octopus tutorial](https://octopus-code.org/documentation/main/tutorial/model/1d_helium/)（同じソフトクーロン規約） |
| 第一量子化 FEDVR Hartree mean field（[`He_fedvr_GS.cpp`](He_fedvr_GS.cpp)） | **-2.224120** | $L = 20$ , $N_e = 10$ , $n = 8$ , $d\tau = 0.005$ , 1949 itr |
| 第二量子化 FEDVR RHF 虚時間 RK4（[`He_fedvr_HF.cpp`](He_fedvr_HF.cpp)） | **-2.22411968** | 1949 itr で収束 |
| 第二量子化 FEDVR RHF Roothaan-Hall SCF（[`He_fedvr_HF.cpp`](He_fedvr_HF.cpp)） | **-2.22411984** | わずか **12 SCF iter** で収束、HOMO $\varepsilon_0 = -0.7502$ Ha |
| 参考: FD (Hartree mean field) | -2.22837 | [`../FD/`](../FD) |

3 通りの実装が $\sim 10^{-7}$ Ha レベルで一致しており、閉殻 2 電子では Hartree（自己相互作用込み）と RHF（交換項込み）が形式的に一致することも数値的に確認できている。FCI からの差 $\sim 0.014$ Ha は相関エネルギーで、これは Hartree/RHF 近似の系統誤差として妥当。

1D ソフトクーロン Be 原子（ $Z = 4$ , $N_e = 4$ , $N_{\rm occ} = 2$ ）の閉殻 RHF（[`Be_fedvr_HF.cpp`](Be_fedvr_HF.cpp)）：

| 解法 | $E_{\rm RHF}$ [a.u.] | 反復数 |
|---|---|---|
| 虚時間 RK4 | -6.73934282 | 10272 |
| Roothaan-Hall SCF | -6.73934450 | **14** |

両解法の差は $\sim 10^{-6}$ Ha。SCF 対角化はわずか 14 反復で収束し、虚時間 RK4 の効率の悪さが顕著に見える（ $N_{\rm occ}$ が増えると高次の振動モードが入って RK4 の収束が遅くなる）。Be は閉殻 He と違い **多軌道で交換項 $K$ が非自明な役割を持つ**最初の例で、 $\varepsilon_0 = -1.371$ , $\varepsilon_1 = -0.313$ の 2 占有軌道と HOMO/LUMO ギャップ $0.339$ Ha が得られる。

H 原子（ $Z = 1$ ）では FEDVR の直接対角化と虚時間 RK4 が $E_0 = -0.66976685346$ で
差 $\sim 10^{-11}$ で一致することを確認済み（[`H_fedvr_GS.cpp`](H_fedvr_GS.cpp) で実装）。
Hamiltonian の対称性 $\max \lvert H - H^T \rvert \le 2.2 \times 10^{-16}$ も確認済み。

## 10. 参考文献

- T. N. Rescigno and C. W. McCurdy, "Numerical grid methods for quantum-mechanical scattering problems",
  *Phys. Rev. A* **62**, 032706 (2000).
- B. I. Schneider et al., "FEDVR: A high-order discrete variable representation for...", various TDSE/TDDFT papers.
- Octopus 1D helium tutorial: <https://octopus-code.org/documentation/main/tutorial/model/1d_helium/>

---

## 11. 第二量子化での RHF 定式化（[`hf_fedvr.hpp`](hf_fedvr.hpp), [`He_fedvr_HF.cpp`](He_fedvr_HF.cpp)）

[`He_fedvr.hpp`](He_fedvr.hpp) 系列が **第一量子化での Hartree 平均場** （波動関数 $\psi(x)$ を直接持って虚時間で進める）であるのに対し、[`hf_fedvr.hpp`](hf_fedvr.hpp) は **第二量子化での閉殻 RHF（Restricted Hartree-Fock）** を実装する。閉殻 He では両者は数値的に一致するが、後続の post-HF 法（MP2, CI, CCSD, VQE）や多電子系（Li, Be, ...）への拡張・量子コンピュータ実装の出発点として、こちらの定式化が使われる。

### 11.1 積分テンソル（[`fedvr_basis.hpp`](fedvr_basis.hpp) に追加）

第二量子化のハミルトニアンは

$$ \hat{H} = \sum_{p,q,\sigma} h_{pq}\, \hat{c}^\dagger_{p\sigma} \hat{c}_{q\sigma} + \tfrac12 \sum_{p,q,r,s,\sigma\sigma'} V_{pqrs}\, \hat{c}^\dagger_{p\sigma} \hat{c}^\dagger_{q\sigma'} \hat{c}_{s\sigma'} \hat{c}_{r\sigma} $$

DVR の対角性により：

- 1電子積分（疎・バンド幅 $\sim 2n+1$ ）：

$$ h_{pq} = T_{pq} + V_{\rm ne}(x_p)\, \delta_{pq}, \qquad V_{\rm ne}(x) = -\frac{Z}{\sqrt{x^2 + 1}} $$

- 2電子積分は 4 添字が **2 添字に縮約**：

$$ V_{pqrs} = \delta_{pr}\, \delta_{qs}\, V_{pq}, \qquad V_{pq} = w(x_p, x_q) = \frac{1}{\sqrt{(x_p - x_q)^2 + 1}} $$

これが FEDVR が量子化学で重宝される最大の理由（4 添字テンソル $\sim N^4$ → 2 添字 $\sim N^2$）。`build_h_pq(grid, Z)` と `build_V_pq(grid)` で構築する。

### 11.2 閉殻 RHF Fock 行列の DVR 形

閉殻 RHF の Fock 行列を DVR 表現に書き下すと、Coulomb 部は **対角**、Exchange 部は **単純な要素積** という非常に簡潔な形になる：

$$ F_{pq} = h_{pq} + \delta_{pq}\, J_p - \tfrac12\, V_{pq}\, P_{pq} $$

ここで

$$ J_p = \sum_q V_{pq}\, P_{qq}, \qquad P_{pq} = 2 \sum_{i \in \mathrm{occ}} C_{pi}\, C_{qi} $$

総エネルギーは

$$ E_{\rm RHF} = \tfrac12 \sum_{pq} P_{pq} (h_{pq} + F_{pq}) $$

### 11.3 2 つの解法（同じ Fock 表式から）

**Solver A : 虚時間 RK4 + QR 直交化**

$$ \partial_\tau \mathbf{C}_{\rm occ} = -\hat{F}[\mathbf{C}_{\rm occ}]\, \mathbf{C}_{\rm occ} $$

を 4 段 RK4 で進め、毎ステップ QR 分解で軌道を再直交化する。`H_fedvr.hpp` ・ `He_fedvr.hpp` と同じ「虚時間プロパゲーション」スタイル。He（占有 1 軌道）では実質的に L²正規化と等価。

**Solver B : Roothaan-Hall SCF**

$$ \hat{F}[\mathbf{C}_{\rm occ}]\, \mathbf{C} = \mathbf{C}\, \boldsymbol{\varepsilon} $$

DVR では基底重なり行列 $S = I$ なので、Fock 行列をそのまま対称固有値問題として対角化し、最低 $N_{\rm occ}$ 個の固有ベクトルを次の占有軌道として採用する。**わずか 12 反復で収束**（虚時間 RK4 の 1949 反復に対し圧倒的に速い）。

### 11.4 検証結果

`L=20, N_e=10, n=8, Z=2, N_{\rm occ}=1`（閉殻 He）で実行した結果：

| 解法 | $E_{\rm RHF}$ [a.u.] | 反復数 |
|---|---|---|
| Solver A: 虚時間 RK4 | -2.22411968 | 1949 |
| Solver B: Roothaan-Hall SCF | -2.22411984 | 12 |
| 参考：第一量子化 Hartree | -2.224120 | 1949 |

3 つの実装の差は $\sim 10^{-7}$ Ha 以下。Solver B は HOMO $\varepsilon_0 = -0.7502$ Ha も同時に出力する（HF 一電子軌道エネルギー、Koopmans の定理によりイオン化ポテンシャルの近似）。

### 11.5 拡張のロードマップ

`hf_fedvr.hpp` の `RHF_SCFResult` は全 MO 係数行列 $C_{\rm full}$ と軌道エネルギー $\varepsilon$ を返すので、ここを起点に：

- **MP2**：仮想軌道との 2 励起 → 相関エネルギー補正 $E_{\rm MP2}$
- **CISD / FCI**：He 2 電子では CISD = FCI = 約 -2.238 Ha（厳密）
- **量子コンピュータ向け**：Jordan-Wigner で qubit Hamiltonian に変換 → VQE / QPE

すべて $h_{pq}, V_{pq}$ と $C_{\rm full}$ を入力に取る形で実装可能。

---

## 12. 多軌道閉殻 RHF：1D Be 原子（[`Be_fedvr_HF.cpp`](Be_fedvr_HF.cpp)）

He（$N_{\rm occ} = 1$ ）では占有軌道が 1 つしかないため、RHF Fock 行列の交換項 $K_{pq} = V_{pq} P_{pq}$ がクーロン項 $J_p$ の半分を相殺する形で、Hartree-only 平均場と同じエネルギーになる（§9 で確認）。**Be 原子（ $Z = 4$ , $N_e = 4$ , $N_{\rm occ} = 2$ ）** ではこの偶然のキャンセルが起こらず、交換項が独立した寄与を持つ最初のケース。

### 12.1 何が変わるか

- 占有軌道が 2 つ：偶パリティ（1s 様）と奇パリティ（2p 様）
- 密度行列：

$$ P_{pq} = 2 \sum_{i = 1}^{2} C_{pi}\, C_{qi} = 2\, (C_{1p} C_{1q} + C_{2p} C_{2q}) $$

  → He の単一軌道密度 $P = 2 c c^T$ の自然な拡張
- 交換項 $K_{pq}$ は両軌道からの寄与を含み、対角部分以外にも非自明な値を持つ
- HF 軌道エネルギー $\varepsilon_i$ が複数得られ、HOMO/LUMO ギャップが定義できる

### 12.2 実装

`hf_fedvr.hpp` のソルバーは **そのまま流用**できる： `N_occ = 2` で初期推測を 2 軌道分（偶 $\propto e^{-x^2}$ + 奇 $\propto x e^{-x^2}$ ）渡すだけ。QR 直交化が解の収束過程で自動的に正しい 2 軌道を見つける。

### 12.3 数値結果（ $L = 20$ , $N_e = 10$ , $n = 8$ ）

| 量 | 値 |
|---|---|
| $E_{\rm RHF}$ （SCF） | $-6.73934450$ Ha |
| $E_{\rm RHF}$ （虚時間） | $-6.73934282$ Ha |
| $\varepsilon_0$ （1s 様、占有） | $-1.370596$ Ha |
| $\varepsilon_1$ （2p 様、占有） | $-0.312922$ Ha |
| $\varepsilon_2$ （LUMO） | $+0.025806$ Ha |
| HOMO–LUMO ギャップ | $0.338728$ Ha |
| SCF 反復数 | **14** |
| 虚時間 RK4 反復数 | 10272 |

SCF 対角化が **虚時間 RK4 の 700 倍以上速い**。 $N_{\rm occ}$ が増えると、RK4 で解く線形系の条件数が悪化して虚時間収束が遅くなることが顕著に見える（一般的傾向）。

### 12.4 物理的解釈

- 1s 様軌道（ $\varepsilon_0 = -1.37$ ）：核近傍に局在、偶パリティ
- 2p 様軌道（ $\varepsilon_1 = -0.31$ ）：節を持つ奇パリティ、より広がる
- 1D ソフトクーロンモデルでは「角運動量 $\ell$ 」がないので、3D の 1s/2s/2p を「偶/奇パリティ × 動径節の数」で分類する必要がある
- 4 電子全部を 1s に詰めることはできない（Pauli 原理）→ 第 2 占有軌道は 2p 様パリティで現れる

### 12.5 次のステップ

- **Phase C** : 多核（H₂ 分子）への拡張で `build_h_pq` を一般化
- **Phase B** : UHF 拡張で開殻 Li, H 原子に対応
- いずれも `hf_fedvr.hpp` の現在の RHF 実装と並列に追加できる
