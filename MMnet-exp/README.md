#### MMNet experiments：

- Based projrct: [MMnet](https://github.com/hyperconnect/MMNet).

- Training protocol:

  - Datasets: [Deep Automatic Portrait Matting](http://xiaoyongshen.me/webpages/webpage_automatting/).
  - Training step splits into: 304500, 887000 and 1898000 steps which is increment training based 304500 step.
  - Others setting follow the based project.

- Training record. can be found in: [304500-exp](), [887000-exp](), [1898000-exp]()

- Quantitative Overview:

  | **On** **MMNet** **test datasets** | **Train:** **MMNet** **dataset, step=304500** | **Incremental** **train:** **MMNet** **dataset +** **aisegment**, step=887000 | **Train:** **MMNet** **dataset, step=1898000**      **304500+**1593500 |
  | ---------------------------------- | --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
  | MAD   (256x256)                    | 0.0284169                                     | 0.0197250                                                    | 0.02617092                                                   |
  | GAUSS_GRAD   (256x256)             | 0.0046924                                     | 0.0046534                                                    | 0.00426231                                                   |
  | MAD   (800x600)                    | 0.0289083                                     | 0.0202761                                                    | **0.02685847**                                               |
  | GAUSS_GRAD   (800x600)             | 0.0030198                                     | 0.0029478                                                    | **0.00285657**                                               |

  The 1898000 is mostly close to paper results where   **MAD   (800x600)  = 0.0248 and GAUSS_GRAD   (800x600)   = 0.0293 **.

- TensorBoard overview:

  - ***304500 step:***

    ![304500-Gradient](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/304500-Gradient.png)

    ![304500-SAD](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/304500-SAD.png)

    ![304500-loss1](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/304500-loss1.png)

    ![304500-loss2](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/304500-loss2.png)

  - ***887000 step:***

    ![887000-Gradient](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/887000-Gradient.png)

    ![887000-SAD](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/887000-SAD.png)

    ![887000-loss1](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/887000-loss1.png)

    ![887000-loss2](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/887000-loss2.png)

    

  - ***1898000 step:***

    ![1898000-Gradient](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/1898000-Gradient.png)

    ![1898000-SAD](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/1898000-SAD.png)

    ![1898000-loss1](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/1898000-loss1.png)

    ![1898000-loss2](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/1898000-loss2.png)

- Qualitative example fo ***1898000 step***:

  ![img_00001](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/img_00001.jpg)

  ![img_00007](/Users/juphoon/Desktop/erichym/github-pro/MMNet-exp/pic/img_00007.jpg)
