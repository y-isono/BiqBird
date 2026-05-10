# 1-D atom simulation

ソフトクーロン1次元モデルにおける、H原子・He原子（閉殻 Hartree平均場）の電子基底状態を、
有限差分（FD）法と有限要素離散変数表現（FEDVR）法でそれぞれ計算するサンプル群。

時間依存版（外部レーザー場 $E(t)\, x$ を含むTDSE）への拡張も視野に入れた設計。

## ディレクトリ

| | 説明 |
|---|---|
| [`FD/`](FD)       | 有限差分（2次中心差分 + 虚時間 RK4）の実装 |
| [`FEDVR/`](FEDVR) | FEDVR（有限要素 + Gauss–Lobatto DVR + 虚時間 RK4）の実装 |

---

## ハミルトニアン定義（共通規約）

すべて原子単位、ソフトクーロンパラメータ $a = 1$ で固定。

### H 原子（1電子）

$$ i \frac{\partial}{\partial t}\psi(x,t) = \hat{H}\psi(x,t) $$

$$ \hat{H} = -\frac{1}{2}\frac{d^2}{dx^2} - \frac{1}{\sqrt{x^2+1}} + E(t) x $$

### He 原子（2電子・閉殻、Hartree平均場）

$$ \hat{H} = \hat{h}(t) + \hat{W}_{\rho} $$

$$ \hat{h}(t) = -\frac{1}{2}\frac{d^2}{dx^2} - \frac{2}{\sqrt{x^2+1}} + E(t) x $$

$$ \hat{W}_{\rho}(x) = \int \frac{\rho(x')}{\sqrt{(x-x')^2 + 1}} dx' $$

$$ \rho(x) = |\psi(x)|^2 $$

エネルギー期待値は

$$ E[\psi] = 2 \langle\psi|\hat{h}|\psi\rangle + \iint |\psi(x_1)|^2 \frac{1}{\sqrt{(x_1-x_2)^2+1}} |\psi(x_2)|^2 dx_1 dx_2 $$

### 虚時間緩和（基底状態の求め方）

$t \to -i\tau$ の置換により、

$$ i\frac{d}{dt}\Psi(t) = H_0\Psi(t) \longrightarrow \frac{d}{d\tau}\Psi(-i\tau) = -H_0 \Psi(-i\tau) $$

これを 4 次ルンゲクッタで進め、毎ステップ L²正規化することで基底状態に収束させる。

---

## Reference values（既知文献値）

ソフトクーロン規約 $a = 1$ における基底状態エネルギーの参照値：

| 計算 | $E_{\rm tot}$ [a.u.] | 出典・備考 |
|---|---|---|
| **Exact (FCI)** | **-2.23825730** | [Octopus tutorial - 1D helium](https://octopus-code.org/documentation/main/tutorial/model/1d_helium/)（同じソフトクーロン規約、box L=8、grid spacing 0.1）|
| Hartree mean field（厳密解への近似） | $\approx -2.224$ | 単一空間軌道 + Hartree（交換項なし）。FCI から相関エネルギー $\sim 0.014$ Ha 分だけ高い |

## Implementation results（本リポジトリの実装による）

各実装で **RHF SCF**（Roothaan-Hall 直接対角化）を解いた結果。第一量子化スタイル虚時間 RK4・第二量子化虚時間 RK4 はそれぞれ同じ値（収束限界の範囲内）に集約することを各サブディレクトリの README で確認している。

### He 原子（$Z=2$, $N_{\rm occ}=1$, $x_{\rm range}=20$）

| 実装 | パラメータ | DOF | $E_{\rm RHF}$ [Ha] | 反復 |
|---|---|---|---|---|
| FD ([`FD/He_fd_HF.cpp`](FD/He_fd_HF.cpp))           | $\Delta x = 0.4$ | 99 | $-2.22893775$ | 12 |
| FEDVR ([`FEDVR/He_fedvr_HF.cpp`](FEDVR/He_fedvr_HF.cpp)) | $N_e = 10$, $n = 8$ | 79 | $-2.22411984$ | 12 |

両者とも閉殻 RHF 解で、相互に $\sim 0.005$ Ha 異なるが、これは離散化誤差。次節の連続極限スキャンで確認できる通り、両者を細かくしていけば共通の連続極限値（$\sim -2.2242$ Ha）に収束する。

参照値（[Reference values](#reference-values既知文献値) 参照）：FCI（$-2.23826$ Ha）との差 $\sim 0.014$ Ha は RHF 近似の **相関エネルギー**。

### Be 原子（$Z=4$, $N_{\rm occ}=2$, $x_{\rm range}=20$）

| 実装 | パラメータ | DOF | $E_{\rm RHF}$ [Ha] | 反復 |
|---|---|---|---|---|
| FD ([`FD/Be_fd_HF.cpp`](FD/Be_fd_HF.cpp))           | $\Delta x = 0.2$ | 199 | $-6.74646782$ | 17 |
| FEDVR ([`FEDVR/Be_fedvr_HF.cpp`](FEDVR/Be_fedvr_HF.cpp)) | $N_e = 10$, $n = 8$ | 79 | $-6.73934450$ | 16 |

### H₂ 分子（核 2 個、$Z_a=Z_b=1$, 閉殻 RHF, $N_{\rm occ}=1$, $x_{\rm range}=20$）

核間距離 $R$ をスキャンしてポテンシャル曲線 $E_{\rm tot}(R) = E_{\rm RHF}(R) + E_{\rm nn}(R)$ を求める。  
全エネルギーは平衡距離 $R_e \approx 1.6$ a.u. で最小：

| 実装 | パラメータ | $R_e$ [a.u.] | $E_{\rm tot}(R_e)$ [Ha] | $E_{\rm tot}(R=6)$ [Ha] |
|---|---|---|---|---|
| FD ([`FD/H2_fd_HF.cpp`](FD/H2_fd_HF.cpp))               | $\Delta x = 0.2$ | $\approx 1.6$ | $-1.42576$ | $-1.12084$ |
| FEDVR ([`FEDVR/H2_fedvr_HF.cpp`](FEDVR/H2_fedvr_HF.cpp)) | $N_e=10$, $n=8$ | $\approx 1.6$ | $-1.42507$ | $-1.12053$ |

両者の差は全 $R$ で $\sim 5\times10^{-4}$ Ha（離散化誤差レベル）。$R \to \infty$ で RHF が真の解離極限 $2 E_0^{\rm H} = -1.33954$ Ha に到達せず（$R=6$ で $-1.12$ Ha 止まり）、典型的な **RHF 解離問題** が観察できる。詳細・全 14 点のスキャン結果は [`FEDVR/README.md`](FEDVR/README.md) §13 を参照。

#### 整合性チェック（平衡距離 $R = 1.6$ a.u. での 3 解法集約）

[`H2_consistency_check.cpp`](H2_consistency_check.cpp) で平衡距離 $R = 1.6$ における (A) 第一量子化 Hartree+exchange 虚時間、(B) 第二量子化 RHF 虚時間、(C) 第二量子化 RHF SCF の集約を確認できる。He / Be で確認した数学的等価性が **多核外場でも維持される** こと（つまり多核 `build_h_pq` の実装に重大なバグがないこと）の検証。

| | 解法 | $E_{\rm RHF}$ [Ha] | 反復 |
|---|---|---|---|
| FD ($\Delta x=0.2$, $N=199$) | (A) 第一量子化 Hartree+exchange 虚時間 | $-1.95575566606$ | 2878 |
| FD | (B) 第二量子化 RHF 虚時間 | $-1.95575566606$ | 2878 |
| FD | (C) 第二量子化 RHF SCF | $-1.95575678865$ | 16 |
| FEDVR ($N_e=10$, $n=8$, $N=79$) | (B) 第二量子化 RHF 虚時間 | $-1.95506731179$ | 3009 |
| FEDVR | (C) 第二量子化 RHF SCF | $-1.95506843152$ | 13 |

各離散化内で **3 解法（FD）/ 2 解法（FEDVR）が $\sim 1\times10^{-6}$ Ha で集約**。FEDVR は第一量子化スタイルの多軌道ヘルパーを持たないので (A) は省略（Be も同様の事情）。

```bash
mkdir -p out
EIGEN=/opt/homebrew/include/eigen3
g++ -std=c++17 -O2 -I"$EIGEN" -I"FD" -I"FEDVR" H2_consistency_check.cpp -o out/h2_consistency_check
./out/h2_consistency_check
```

---

## Continuum-limit consistency check

FD（$\Delta x \to 0$）と FEDVR（$n_{\rm order} \to$ 大）は **異なる離散化スキーム**だが、両者を細かくしていけば **共通の連続極限値** に収束するはず。これを実機で確認するスクリプトが [`continuum_limit_scan.cpp`](continuum_limit_scan.cpp)。各設定で `fedvr::rhf_scf` を回し、得られた SCF エネルギーを比較する。

ビルド・実行：

```bash
mkdir -p out
EIGEN=/opt/homebrew/include/eigen3
g++ -std=c++17 -O2 -I"$EIGEN" -I"FD" -I"FEDVR" continuum_limit_scan.cpp -o out/continuum_limit_scan
./out/continuum_limit_scan
```

### He（$x_{\rm range}=20$）の結果

| scheme | パラメータ | DOF | $E_{\rm RHF}$ [Ha] |
|---|---|---|---|
| FD     | $\Delta x = 0.4$    | 99   | $-2.22893775$ |
| FD     | $\Delta x = 0.2$    | 199  | $-2.22537408$ |
| FD     | $\Delta x = 0.1$    | 399  | $-2.22449966$ |
| FD     | $\Delta x = 0.05$   | 799  | $-2.22428201$ |
| FD     | $\Delta x = 0.025$  | 1599 | $-2.22422766$ |
| FEDVR  | $N_e=10, n=4$       | 39   | $-2.20928079$ |
| FEDVR  | $N_e=10, n=6$       | 59   | $-2.22526032$ |
| FEDVR  | $N_e=10, n=8$       | 79   | $-2.22411984$ |
| FEDVR  | $N_e=10, n=10$      | 99   | $-2.22421263$ |
| FEDVR  | $N_e=10, n=12$      | 119  | $-2.22420920$ |

→ FD は **下から**（過大評価、$O(\Delta x^2)$ で減衰）、FEDVR は **指数的**に収束し、両者は連続極限値 $\boxed{E_{\rm RHF}^{\rm He} \approx -2.22421 \text{ Ha}}$ で一致。最も細かい設定（FD $\Delta x=0.025$ vs FEDVR $n=12$）の差は $\sim 2\times10^{-5}$ Ha。

### Be（$x_{\rm range}=20$）の結果

| scheme | パラメータ | DOF | $E_{\rm RHF}$ [Ha] |
|---|---|---|---|
| FD     | $\Delta x = 0.2$    | 199  | $-6.74646782$ |
| FD     | $\Delta x = 0.1$    | 399  | $-6.74119375$ |
| FD     | $\Delta x = 0.05$   | 799  | $-6.73988501$ |
| FD     | $\Delta x = 0.025$  | 1599 | $-6.73955842$ |
| FEDVR  | $N_e=10, n=4$       | 39   | $-6.76854572$ |
| FEDVR  | $N_e=10, n=6$       | 59   | $-6.74054447$ |
| FEDVR  | $N_e=10, n=8$       | 79   | $-6.73934450$ |
| FEDVR  | $N_e=10, n=10$      | 99   | $-6.73944845$ |
| FEDVR  | $N_e=10, n=12$      | 119  | $-6.73944868$ |

→ He と同じ傾向で、両者は $\boxed{E_{\rm RHF}^{\rm Be} \approx -6.7394 \text{ Ha}}$ あたりで一致（差 $\sim 10^{-4}$ Ha）。

### 解釈

- **FD は連続極限値より深い** ところに位置し、$\Delta x \to 0$ で上から漸近。これは FD 2 次中心差分の運動エネルギー演算子が連続 Laplacian の固有値を過小評価する性質に由来する（基底関数を陽に持たないため、有限基底変分原理を厳密には満たさない）。
- **FEDVR は連続極限値の上下に振動的**に近づく（$n=4$ では浅め、$n=6$ で行き過ぎ、$n=8\to 12$ で正しい値に収束）。これは GLL 求積誤差と基底空間サイズのバランスで生じる典型挙動。
- 両者が同じ連続極限値に集約することで、コード両系統に重大なバグがないことが確認できる。

---

## H 原子の参考値

| | $E_0$ [a.u.] |
|---|---|
| 直接対角化（ $L = 20$ , $N_e = 10$ , $n = 8$ , FEDVR） | -0.66976685 |
| FEDVR 虚時間 RK4 | -0.66976685（差 $\sim 10^{-11}$ ）|

文献値は $a = 1$ のソフトクーロン水素で $E_0 \approx -0.6699$ a.u. として知られている。
