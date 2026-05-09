# 1-D atom simulation

ソフトクーロン1次元モデルにおける、H原子・He原子（閉殻 Hartree平均場）の電子基底状態を、
有限差分（FD）法と有限要素離散変数表現（FEDVR）法でそれぞれ計算するサンプル群。

時間依存版（外部レーザー場 $E(t)x$ を含むTDSE）への拡張も視野に入れた設計。

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

これを4次ルンゲクッタで進め、毎ステップ L²正規化することで基底状態に収束させる。

---

## Reference values（既知文献値）

ソフトクーロン規約 $a=1$ における基底状態エネルギーの参照値：

| 計算 | $E_{\rm tot}$ [a.u.] | 出典・備考 |
|---|---|---|
| **Exact (FCI)** | **-2.23825730** | [Octopus tutorial - 1D helium](https://octopus-code.org/documentation/main/tutorial/model/1d_helium/)（同じソフトクーロン規約、box L=8、grid spacing 0.1）|
| Hartree mean field（厳密解への近似） | $\approx -2.224$ | 単一空間軌道 + Hartree（交換項なし）。FCI から相関エネルギー $\sim 0.014$ Ha 分だけ高い |

## Implementation results（本リポジトリの実装による）

| 実装 | $E_{\rm tot}$ | $E_1$ | $E_2$ | 反復回数 |
|---|---|---|---|---|
| FD ([`FD/He_GS.cpp`](FD/He_GS.cpp))           | -2.22837 | -2.94721 | 0.71884 | 239 |
| FEDVR ([`FEDVR/He_fedvr_GS.cpp`](FEDVR/He_fedvr_GS.cpp)) | -2.22412 | -2.94774 | 0.72362 | 1949 |

両者とも Hartree 平均場近似であり、相互に $\sim 0.004$ Ha の精度で一致。
FCI の $-2.23826$ Ha との差 $\sim 0.014$ Ha は **相関エネルギー** に相当し、これは Hartree 近似の系統誤差として妥当。

## H 原子の参考値

| | $E_0$ [a.u.] |
|---|---|
| 直接対角化 ($L=20$, $N_e=10$, $n=8$, FEDVR) | -0.66976685 |
| FEDVR 虚時間 RK4 | -0.66976685（差 ~ $10^{-11}$）|

文献値は $a=1$ のソフトクーロン水素で $E_0 \approx -0.6699$ a.u. として知られている。
