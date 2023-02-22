import pprint

from learner import get_args, train_agent, watch


def main(args=get_args()):
    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    # assert result["best_reward"] >= args.win_rate

    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, agent)


if __name__ == "__main__":
    main(get_args())