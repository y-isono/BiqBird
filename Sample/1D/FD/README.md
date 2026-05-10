# 1-D H/He/Be atom — Finite Difference (FD) implementation

有限差分法（2次中心差分）による空間離散化と虚時間 RK4 緩和で、
1次元ソフトクーロン H / He / Be 原子の電子基底状態を求める実装。

物理ハミルトニアン（共通定義）は親ディレクトリ
[`../README.md`](../README.md) を参照。
FEDVR 版（[`../FEDVR/`](../FEDVR/)）と完全に並行する構造になっている。

旧コード（複素数ベースの単一軌道専用 Hartree 実装）は [`legacy/`](legacy/) に保存。

---

## 1. ファイル一覧

| ファイル | 内容 |
|---|---|
| `fd_basis.hpp` | `FDGrid` 構造体（等間隔グリッド + 運動行列 $T$ ）と第二量子化用の積分テンソル `build_h_pq` 、 `build_V_pq` |
| `hartree_fd.hpp` | 第一量子化スタイルのソルバー：単一軌道 Hartree 平均場（He 用） + 多軌道 RHF（Be 用） |
| `H_GS.cpp` | H 原子（ $Z = 1$ ）の基底状態、直接対角化 + 虚時間 RK4 |
| `He_GS.cpp` | 閉殻 He（ $Z = 2$ , $N_{\rm occ} = 1$ ）の Hartree 平均場、第一量子化 |
| `He_fd_HF.cpp` | 閉殻 He の RHF、第二量子化（FEDVR の [`hf_fedvr.hpp`](../FEDVR/hf_fedvr.hpp) を再利用） |
| `Be_GS.cpp` | 閉殻 Be（ $Z = 4$ , $N_{\rm occ} = 2$ ）の RHF、第一量子化スタイル |
| `Be_fd_HF.cpp` | 閉殻 Be の RHF、第二量子化（FEDVR の [`hf_fedvr.hpp`](../FEDVR/hf_fedvr.hpp) を再利用） |
| `H2_fd_HF.cpp` | 閉殻 H₂ 分子（核 2 個、$N_e = 2$ , $N_{\rm occ} = 1$ ）の RHF ポテンシャル曲線スキャン |
| `legacy/` | 旧実装（[`legacy/README.md`](legacy/README.md) 参照） |

---

## 2. 設計指針

- **実数で書く**：基底状態のハミルトニアンは実対称、固有関数は実数で取れる。
  `Eigen::VectorXd` / `Eigen::SparseMatrix<double>` を使う。TDSE への拡張は別ファイルで `Xcd` 化する想定。
- **FEDVR 側との対称性**：`fd_basis.hpp` の API は [`fedvr_basis.hpp`](../FEDVR/fedvr_basis.hpp) と同じシグネチャを持つので、FEDVR 用に書いた閉殻 RHF ソルバー（`fedvr::rhf_imag_time`, `fedvr::rhf_scf`）を **そのまま FD でも使える**。
- **DVR-like FD 基底**：等間隔グリッド + 一定重み $w_p = \Delta x$ を「形式的な DVR 基底」とみなすことで、FEDVR と同じ閉形式（ $V_{pqrs} = \delta_{pr}\delta_{qs} V_{pq}$ ）が成立する。

---

## 3. ハミルトニアン

H ：

$$ \hat{H}_{\rm H} = -\frac{1}{2}\frac{d^2}{dx^2} - \frac{1}{\sqrt{x^2+1}} $$

He（閉殻）：

$$ E[\psi] = 2 \langle\psi|\hat{h}_2|\psi\rangle + \iint |\psi(x_1)|^2 w(x_1, x_2) |\psi(x_2)|^2 dx_1 dx_2 $$

Be（閉殻 RHF、 $N_{\rm occ} = 2$ ）：

$$ F_{pq} = h_{pq} + \delta_{pq}\, J_p - \tfrac12\, V_{pq}\, P_{pq} $$

$$ J_p = \sum_q V_{pq}\, P_{qq}, \qquad P_{pq} = 2 \sum_{i \in \mathrm{occ}} C_{pi}\, C_{qi} $$

詳細は [`../FEDVR/README.md`](../FEDVR/README.md) §11 を参照。

---

## 4. 数値手法

- 空間離散化：等間隔グリッド、2次中心差分（運動行列 $T$ は対称三重対角の疎行列）
- 時間発展：虚時間 $t \to -i\tau$ → $\partial_\tau\, c = -F[c]\, c$ を **RK4** で進行
- 多軌道では各 RK4 ステップ後に QR 直交化（DVR 重なりは $S = I$ ）
- SCF 対角化：Fock 行列を Eigen `SelfAdjointEigenSolver` で対角化、最低 $N_{\rm occ}$ 個の固有ベクトルを採用

---

## 5. ビルド・実行例

```bash
mkdir -p out
EIGEN=/opt/homebrew/include/eigen3
g++ -std=c++17 -O2 -I"$EIGEN" H_GS.cpp     -o out/h_gs    && ./out/h_gs
g++ -std=c++17 -O2 -I"$EIGEN" He_GS.cpp    -o out/he_gs   && ./out/he_gs
g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" He_fd_HF.cpp -o out/he_fd_hf && ./out/he_fd_hf
g++ -std=c++17 -O2 -I"$EIGEN" Be_GS.cpp    -o out/be_gs   && ./out/be_gs
g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" Be_fd_HF.cpp -o out/be_fd_hf && ./out/be_fd_hf
g++ -std=c++17 -O2 -I"$EIGEN" -I"../FEDVR" H2_fd_HF.cpp -o out/h2_fd_hf && ./out/h2_fd_hf > out/h2_fd_hf.csv
```

`-I"../FEDVR"` は第二量子化 RHF ソルバー [`../FEDVR/hf_fedvr.hpp`](../FEDVR/hf_fedvr.hpp) を読むため。

---

## 6. 既定パラメータ

| | $x_{\rm range}$ | $\Delta x$ | $N$ (DOF) | $d\tau$ | max_itr |
|---|---|---|---|---|---|
| H  | 15  | 0.2 | 149 | 0.05  | 50000  |
| He | 20  | 0.4 |  99 | 0.01  | 30000  |
| Be | 20  | 0.2 | 199 | 0.002 | 200000 |

DOF $N = \mathrm{round}(2\, x_{\rm range} / \Delta x) - 1$ （両端 Dirichlet）

$d\tau$ は CFL 条件 $d\tau \cdot \lambda_{\max}(T) \lesssim 1$（$\lambda_{\max}(T) \sim 1/\Delta x^2$）の安定範囲に取りつつ、虚時間 RK4 が真の RHF 停留点に到達できるよう十分小さく設定している（履歴：He は以前 $d\tau=0.20$ だったが、エネルギー差ベースの収束判定が早期に発動して SCF 解より $\sim 5\times10^{-4}$ Ha 浅いところで停止する問題があった）。

---

## 7. 検証結果

### H 原子（ $Z = 1$ ）

| 解法 | $E_0$ [a.u.] |
|---|---|
| 直接対角化 | $-0.6701078$ |
| 虚時間 RK4 | $-0.6701078$（差 $\sim 10^{-13}$ ） |

文献値は $-0.6699$ Ha 程度。FD 2次差分のグリッド誤差で少し深め（FEDVR 版は $-0.66977$ で文献値により近い）。

### He 原子（ $Z = 2$ , 閉殻、 $N_{\rm occ} = 1$ ）

3 つの解法はすべて同じ RHF 解に収束する（単一軌道では Hartree-only と RHF が数学的に厳密等価；§3 参照）：

| 計算 | $E_{\rm tot}$ [a.u.] | 反復 | 備考 |
|---|---|---|---|
| (A) FD 第一量子化 Hartree（[`He_GS.cpp`](He_GS.cpp)） | $-2.22893646713$ | 2632 | 単一軌道 Hartree 平均場、虚時間 RK4 |
| (B) FD 第二量子化 RHF 虚時間（[`He_fd_HF.cpp`](He_fd_HF.cpp) Solver A） | $-2.22893646713$ | 2612 | (A) と $3\times10^{-12}$ で完全一致 |
| (C) FD 第二量子化 RHF SCF（[`He_fd_HF.cpp`](He_fd_HF.cpp) Solver B） | $-2.22893774527$ | 12 | Roothaan-Hall 直接対角化（厳密な停留点） |
| 参考: FEDVR RHF SCF | $-2.22411965$ | 14 | [`../FEDVR/He_fedvr_HF.cpp`](../FEDVR/He_fedvr_HF.cpp) |
| 参考: Exact (FCI, Octopus) | $-2.238257$ | — | 1D ソフトクーロン He の厳密解 |

(A)(B) と (C) のあいだに残る $\sim 1.3\times10^{-6}$ Ha の差は、虚時間 RK4 の有限時間ステップによる収束限界（$d\tau$ をさらに絞ればさらに縮小）。**3 解法が共通の RHF 値に集約しており、第一量子化と第二量子化の数学的等価性が数値的にも確認できる**。

FD グリッドは粗いので FEDVR より $\sim 0.005$ Ha 深い結果になるが、これは FD 2 次差分の運動エネルギー演算子の離散化誤差（$\Delta x \to 0$ で約 $-2.2243$ Ha に収束、[`He_fd_HF_dx_scan.cpp`](He_fd_HF_dx_scan.cpp) の出力参照）。

### Be 原子（ $Z = 4$ , 閉殻、 $N_{\rm occ} = 2$ ）

He と同様、3 解法すべてが同じ RHF 解に収束する：

| 計算 | $E_{\rm RHF}$ [a.u.] | 反復 | 備考 |
|---|---|---|---|
| (A) FD 第一量子化 RHF（[`Be_GS.cpp`](Be_GS.cpp)） | $-6.74646613323$ | 10239 | 多軌道 RHF（Hartree+exchange）、虚時間 RK4 |
| (B) FD 第二量子化 RHF 虚時間（[`Be_fd_HF.cpp`](Be_fd_HF.cpp) Solver A） | $-6.74646613323$ | 10239 | (A) と数値的に完全一致 |
| (C) FD 第二量子化 RHF SCF（[`Be_fd_HF.cpp`](Be_fd_HF.cpp) Solver B） | $-6.74646781531$ | 17 | Roothaan-Hall 直接対角化 |
| HOMO（2占有目） | $-0.31402335$ | — | （SCF版） |
| LUMO（仮想最低） | $+0.02574012$ | — | |
| HOMO-LUMO ギャップ | $+0.33976347$ | — | |
| 参考: FEDVR RHF SCF | $-6.739345$ | 14 | [`../FEDVR/Be_fedvr_HF.cpp`](../FEDVR/Be_fedvr_HF.cpp) |

(A)(B) と (C) のあいだに残る $\sim 1.7\times10^{-6}$ Ha の差はやはり虚時間 RK4 の収束限界。FD は FEDVR より $\sim 7\times10^{-3}$ Ha 深いが、これはグリッドの粗さによる離散化誤差で、$\Delta x$ を細かくすれば FEDVR に近づく（変分原理の方向性自体は破られていない；FEDVR と FD は別の有限基底）。

### H₂ 分子（$Z_a = Z_b = 1$, 閉殻 RHF, $N_{\rm occ} = 1$, $x_{\rm range}=20$, $\Delta x = 0.2$）

[`H2_fd_HF.cpp`](H2_fd_HF.cpp) で核間距離 $R$ を 0.4〜6.0 a.u. でスキャンし、ポテンシャル曲線 $E_{\rm tot}(R) = E_{\rm RHF}(R) + E_{\rm nn}(R)$ を CSV 出力する。

| $R$ [a.u.] | $E_{\rm RHF}$ [Ha] | $E_{\rm nn}$ [Ha] | $E_{\rm tot}$ [Ha] |
|---|---|---|---|
| 0.4 | $-2.20363$ | $0.92848$ | $-1.27516$ |
| 1.0 | $-2.10215$ | $0.70711$ | $-1.39504$ |
| 1.4 | $-2.00680$ | $0.58124$ | $-1.42556$ |
| **1.6** | $-1.95576$ | $0.53000$ | **$-1.42576$** ← 最低 |
| 2.0 | $-1.85333$ | $0.44721$ | $-1.40611$ |
| 3.0 | $-1.62786$ | $0.31623$ | $-1.31163$ |
| 6.0 | $-1.28524$ | $0.16440$ | $-1.12084$ |

（全 14 点の数値は [`H2_fd_HF.cpp`](H2_fd_HF.cpp) の出力 CSV を参照）

平衡距離 $R_e \approx 1.6$ a.u.、最小エネルギー $E_{\rm tot}(R_e) \approx -1.426$ Ha。FEDVR 版（[`../FEDVR/H2_fedvr_HF.cpp`](../FEDVR/H2_fedvr_HF.cpp)）と全 R で $\sim 5\times10^{-4}$ Ha 以内で並行する結果になる（離散化誤差レベル）。$R \to \infty$ で **RHF が解離極限（$2 E_0^{\rm H} = -1.34$ Ha）に到達できない**様子（$R = 6$ で $-1.12$ Ha 止まり）も見える。これは閉殻 RHF の本質的な限界で、UHF や post-HF への動機付けになる（[`../FEDVR/README.md`](../FEDVR/README.md) §13 で詳細）。

---

## 8. 注意点・既知の挙動

- **CFL 安定性**：FD の運動行列 $T$ は $\lambda_{\max}(T) \sim 1/\Delta x^2$ なので、虚時間 RK4 は $d\tau \cdot \lambda_{\max}(T) \lesssim 1$ を満たす範囲で安定。$\Delta x$ を変える場合は $d\tau \propto \Delta x^2$ でスケールする必要がある（[`He_fd_HF_dx_scan.cpp`](He_fd_HF_dx_scan.cpp) で確認）。
- **虚時間 RK4 の収束判定**：エネルギー差 $|E_n - E_{n-1}| < \text{thresh}$ は **エネルギーが平坦に見えても軌道はまだ収束していない** という早期発動が起こり得る。SCF（直接対角化）と一致させるには $d\tau$ を十分小さく取り、反復回数も多めに設定する。検証コード [`He_fd_HF_diagnostic.cpp`](He_fd_HF_diagnostic.cpp) / [`He_fd_HF_imag_robust.cpp`](He_fd_HF_imag_robust.cpp) を参照。
- **解法間の関係**：単一軌道（He）では Hartree-only と RHF Fock が作用素レベルで厳密等価、多軌道（Be）でも RHF Fock を共通に用いることで第一量子化スタイル虚時間と第二量子化 SCF が同一の停留点に収束する。

---

## 9. 参考文献

- 1D ソフトクーロン水素・ヘリウム：Octopus tutorial、 *Phys. Rev. A* 等の文献
- DVR-like 基底での閉殻 RHF：詳細は [`../FEDVR/README.md`](../FEDVR/README.md) §11–§12 を参照
