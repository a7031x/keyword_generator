import numpy as np
import preprocess
import utils
import config
from data import TrainFeeder, align2d, Dataset


def run_epoch(sess, model, feeder, writer):
    feeder.prepare('dev')
    while not feeder.eof():
        aids, qv, av, kb = feeder.next(32)
        feed = model.feed(aids, qv, av, kb)
        answer_logit, question_logit = sess.run([model.answer_logit, model.question_logit], feed_dict=feed)
        question = [id for id, v in enumerate(question_logit) if v >= 0]
        answer = [id for id, v in enumerate(answer_logit) if v >= 0]
        return question, answer
        

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


class Evaluator(TrainFeeder):
    def __init__(self, dataset=None):
        super(Evaluator, self).__init__(Dataset() if dataset is None else dataset)


    def create_feed(self, answer, question):
        question = question.split(' ')
        answer = answer.split(' ')
        aids = self.asent_to_id(answer)
        qv, _ = self.label_qa(question)
        st = self.seq_tag(question, answer)
        return aids, qv, st, 1.0


    def predict(self, sess, model, answer, question):
        aids, qv, av, kb = self.create_feed(answer, question)
        feed = model.feed([aids], [qv], [av], kb)
        answer_logit, question_logit = sess.run([model.answer_logit, model.question_logit], feed_dict=feed)
        #question_ids = [id for id, v in enumerate(question_logit[0]) if v >= 0]
        #answer_ids = [id for id, v in enumerate(answer_logit[0]) if v >= 0]
        qids = sorted(enumerate(question_logit[0]), key=lambda x:-x[1])[:10]
        aw = [word for word,value in zip(answer.split(' '), answer_logit[0]) if value >= 0]
        qw = self.qids_to_sent([id for id,_ in qids])
        print('==================================================')
        print('question', question)
        print('words', qw, aw)
        print('score', [v for _,v in qids], ['{}:{:>.4f}'.format(w,x) for w,x in zip(answer.split(' '), answer_logit[0])])
        return qw, aw 


    def evaluate(self, sess, model):
        self.prepare('dev')
        aids, qv, st, kb = self.next(64)
        feed = model.feed(aids, qv, st, kb)
        loss = sess.run(model.loss, feed_dict=feed)
        return loss


if __name__ == '__main__':
    from model import Model
    import tensorflow as tf
    model = Model(config.checkpoint_folder)
    evaluator = Evaluator()
    with tf.Session() as sess:
        model.restore(sess)
        #evaluator.evaluate(sess, model, 'The cat sat on the mat', 'what is on the mat')
        evaluator.predict(sess, model,
            '手 账 ， 指 用于 记事 的 本子 。 日本 有 一 个 很 有趣 的 现象 ， 无论 男女老少 ， 都 随身携带 一 个 称为 “ 手帐 ” 的 笔记本 ， 随时随地 就 掏 出来 记 点 什么 。 如果 你 要 和 他 （ 她 ） 约定 一 件 什么 事情 的 时候 ， 对方 一定 先 掏出 手帐 看一下 。 在 通讯 科技 如此 发达 的 日本 ， 却 人人 都 依赖 这样 原始 的 记录 方式 ， 这 不 禁 让人惊讶 。 　　 如果 不是 亲眼所见 ， 外国人 真的很难 相信 “ 手帐 ” 对 日本人 的 重要性 。 手帐 并 不是 　　 单纯 的 备忘录 ， 除了 提醒 自己 一些 家人 、 朋友 、 客户 的 生日 、 约会 ， 更重要 的 是 安排 自己 每天 的 工作 、 生活 ， 兼具 日记 功能 。 手帐 大都 制作精美 ， 带有 日历 和 笔 ， 可以 夹 些 名片 和 纸片 ， 不同 的 页面 划分 具有 超强 的 整理 功能 ， 以 满足 不同 类型 的 需要 。 比如 主妇 专用 手帐 就会 包含 家政 开支 的 内容 等 。 总之 ， 日本 的 手帐 简直 无所不能 。 难怪 小小 的 一 个 本本 ， 让 商家 费 足 了 心思 。 特别 近年来 ， 手帐 热潮 不断 升温 。 除了 在 杂志 中 以 专题 出现 外 ， 更 是 在 作家 笔下 上升 为 “ 手帐 哲学 ” ， 甚至 有 “ 手帐 的 使用方法 决定 你 的 人生 ” 的 说法 。 是否 有些 太夸张 了 ？ 　　 现在 是 日本 的 年底 ， 又 到 了 手帐 热卖 的 季节 。 大大小小 的 百货 、 超市 、 书店 、 便利店 等 都 设 了 专柜 ， 摆 满 了 花花绿绿 的 、 保管 让 你 挑 得 眼花缭乱 的 手帐 。 许多 国际知名 品牌 如 BURBERRY 、 GUCCI 、 PRADAD 等 也 瞄准 这块 市场 ， 纷纷 推出 自家 个性 手帐 。 有 不同 的 大小 、 设计 、 品牌 、 价钱 、 页数 可 供 选择 ， 总 有 一 本 是 你 的 心头好 。 再 有 喜欢 与众不同 的 人 ， 就 只好 自己动手 制作 一 本 绝版 的 手帐 了 。 没有 手帐 ， 新的一年 怎么 开始 ？ 　　 是 日本人 健忘 吗 ？ 应该说 他们 喜欢 按部就班 ， 凡事 都 提前 很久 就 作 好 计划 ， 然后 反复 确认 。 就算 家人 之间 也是 如此 。 这 也许 是 日本人 “ 岛 民 性格 ” 的 一 个 表现 吧 。 处于 随时 都 可能 发生 的 地震 、 台风 的 地理位置 ， 只有 未雨绸缪 地 作 好 防范 ， 晚上 才能 睡 得 安稳 。',
            '手 账 是 什么')
        evaluator.predict(sess, model,
            '为了 对 现阶段 互联网 金融 的 模式 做 一 个 清晰 的 界定 ， 投融贷 P2P 网站 工作 人员 调研 走访 ， 深度 解析 资讯 ， 最终 梳理 出 第三方支付 、 P2P 网贷 、 大 数据 金融 、 众筹 、 信息化 金融机构 、 互联网 金融 门户 等 六大 互联网 金融 模式 。 由于 互联网 金融 正 处于 快速发展 期 ， 目前 的 分类 也 仅仅 是 一 个 阶段 的 粗浅 分类 ， 即使 在 将 电子货币 、 虚拟货币 归入 第三方支付 这 一 模式 之后 ， 六大 模式 也 无法 包容 诸如 比特币 等 新兴 互联网 金融 创新 产物 。 软 交 所 互联网 金融 实验室 一方面 将 持续 研究 互联网 金融 的 最新动态 及 发展趋势 ， 另一方面 也 将 联合 相关 金融投资机构 ， 为 软件 和 信息服务业 企业 提供 更加 丰富 的 投融资 服务 ， 促进 产业 发展 。 模式 1 ： 第三方支付 第三方支付 ( Third - Party Payment ) 狭义 上 是 指 具备 一定 实力 和 信誉 保障 的 非 银行 机构 ， 借助 通信 、 计算机 和 信息安全 技术 ， 采用 与 各 大 银行 签约 的 方式 ， 在 用户 与 银行 支付结算系统 间 建立 连接 的 电子支付 模式 。 根据 央行 2010 年 在 《 非 金融机构 支付 服务 管理办法 》 中 给 出 的 非 金融机构 支付 服务 的 定义 ， 从 广义 上 讲 第三方支付 是 指 非 金融机构 作为 收 、 付款人 的 支付 中介 所 提供 的 网络 支付 、 预付卡 、 银行卡 收 单 以及 中国人民银行 确定 的 其他 支付 服务 。 第三方支付 已 不仅仅 局限 于 最初 的 互联网 支付 ， 而是 成为 线上 线 下 全面 覆盖 ， 应用 场景 更为 丰富 的 综合 支付 工具 。 模式 2 ： P2P 网贷 P2P ( Peer - to - Peer lending ) ， 即 点对点 信贷 。 P2P 网贷 是 指 通过 第三方 互联网 平台 进行 资金 借 、 贷 双方 的 匹配 ， 需要 借贷 的 人群 可以 通过 网站 平台 寻找 到 有 出借 能力 并且 愿意 基于 一定 条件 出借 的 人群 ， 帮助 贷款 人 通过 和 其他 贷款 人 一起 分担 一 笔 借款 额度 来 分散风险 ， 也 帮助 借款人 在 充分 比较 的 信息 中 选择 有 吸引力 的 利率 条件 。 目前 ， 出现 了 2 种 运营模式 ， 一 是 纯 线上 模式 ， 其 特点 是 资金 借贷 活动 都 通过 线上 进行 ， 不 结合 线 下 的 审核 。 通常 这些 企业 采取 的 审核 借款人 资质 的 措施 有 通过 视频 认证 、 查看 银行 流水 账单 、 身份认证 等 。 第 二 种 是 线上 线 下 结合 的 模式 ， 借款人 在线 上 提交 借款 申请 后 ， 平台 通过 所在 城市 的 代理商 采取 入户 调查 的 方式 审核 借款人 的 资 信 、 还款 能力 等 情况 。 模式 3 ： 大 数据 金融 大 数据 金融 是 指 集合 海量 非 结构化 数据 ， 通过 对其 进行 实时 分析 ， 可以 为 互联网 金融机构 提供 客户 全方位 信息 ， 通过 分析 和 挖掘 客户 的 交易 和 消费 信息 掌握 客户 的 消费习惯 ， 并 准确 预测 客户 行为 ， 使 金融机构 和 金融服务平台 在 营销 和 风险控制 方面 有的放矢 。 基于 大 数据 的 金融 服务平台 主要 指 拥有 海量 数据 的 电子商务 企业 开展 的 金融服务 。 大 数据 的 关键 是 从 大量 数据 中 快速 获取 有用 信息 的 能力 ， 或者 是 从 大 数据 资产 中 快速 变现 的 能力 。 因此 ， 大 数据 的 信息处理 往往 以 云计算 为 基础 。 模式 4 ： 众筹 众筹 大意 为 大众 筹资 或 群众 筹资 ， 是 指 用 团购 预购 的 形式 ， 向 网友 募集 项目 资金 的 模式 。 本意 众筹 是 利用互联网 和 SNS 传播 的 特性 ， 让 创业 企业 、 艺术家 或 个人 对 公众 展示 他们 的 创意 及 项目 ， 争取 大家 的 关注 和 支持 ， 进而 获得 所 需要 的 资金 援助 。 众筹 平台 的 运作模式 大同小异 — — 需要 资金 的 个人 或 团队 将 项目策划 交给 众筹 平台 ， 经过 相关 审核 后 ， 便 可以 在 平台 的 网站 上 建立 属于 自己 的 页面 ， 用来 向 公众 介绍 项目 情况 。 模式 5 ： 信息化 金融机构 所谓 信息化 金融机构 ， 是 指 通过 采用 信息技术 ， 对 传统 运营 流程 进行 改造 或 重构 ， 实现 经营 、 管理 全面 电子化 的 银行 、 证券 和 保险 等 金融机构 。 金融 信息化 是 金融业 发展趋势 之 一 ， 而 信息化 金融机构 则 是 金融创新 的 产物 。 从 金融 整个 行业 来看 ， 银行 的 信息化建设 一直 处于 业 内 领先水平 ， 不仅 具有 国际 领先 的 金融 信息技术 平台 ， 建成 了 由 自助银行 、 电话银行 、 手机 银行 和 网上银行 构成 的 电子银行 立体 服务体系 ， 而且 以 信息化 的 大手笔 — — 数据 集中 工程 在 业 内 独领风骚 ， 其 除了 基于 互联网 的 创新 金融服务 之外 ， 还 形成 了 “ 门户 ” “ 网银 、 金融 产品 超市 、 电商 ” 的 一拖 三 的 金融 电商 创新 服务 模式 。 模式 6 ： 互联网 金融 门户 互联网 金融 门户 是 指 利用互联网 进行 金融 产品 的 销售 以及 为 金融 产品 销售 提供 第三方服务 的 平台 。 它 的 核心 就是 “ 搜索 比价 ” 的 模式 ， 采用 金融 产品 垂直 比价 的 方式 ， 将 各 家 金融机构 的 产品 放在 平台 上 ， 用户 通过 对比 挑选 合适 的 金融 产品 。 互联网 金融 门户 多元化 创新发展 ， 形成 了 提供 高端 理财 投资 服务 和 理财产品 的 第三方 理财 机构 ， 提供 保险 产品 咨询 、 比价 、 购买 服务 的 保险 门户网站 等 。 这种 模式 不存在 太多 政策 风险 ， 因为 其 平台 既不 负责 金融 产品 的 实际 销售 ， 也 不 承担 任何 不良 的 风险 ， 同时 资金 也 完全 不 通过 中间 平台 。',
            '哪 种 互联网 金融 出现 问题 最多 ?')