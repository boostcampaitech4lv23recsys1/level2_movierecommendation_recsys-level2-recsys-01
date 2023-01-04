import argparse
import os
from ensembles.ensembles import Ensemble
from datetime import datetime
from pytz import timezone

def main(args):
    filepath = args.FILE_PATH
    savepath = args.RESULT_PATH
    os.makedirs(filepath, exist_ok=True)
    if os.listdir(filepath) == []:
        raise ValueError(f"{filepath}에 csv 파일을 넣어주세요.")
    os.makedirs(savepath, exist_ok=True)

    en = Ensemble(filepath=filepath)

    if args.WEIGHT1 and en.csv_nums != len(args.WEIGHT1[0]):
        raise ValueError("csv 수와 WEIGHT1 입력값의 수가 일치하지 않습니다. 다시 실행해주세요.")

    if args.STRATEGY == "HARD":
        print(f"********** \nSTRATEGY: {args.STRATEGY} \nWeighted1: None \nWeighted2: None \n**********")
        strategy_title = 'H'
        result = en.hard()
    elif args.STRATEGY == "WEIGHTED":
        if args.WEIGHT1 and not args.WEIGHT2:
            print(f"********** \nSTRATEGY: {args.STRATEGY} \nWeighted1: {args.WEIGHT1[0]} \nWeighted2: None \n**********")
            strategy_title = 'W1-'+'-'.join(map(str,*args.WEIGHT1))
            result = en.weighted(*args.WEIGHT1, args.WEIGHT2)
        elif not args.WEIGHT1 and not args.WEIGHT2:
            print("**********")
            print("--WEIGHT1, --WEIGHT2가 입력되지 않아 HARD Voting이 작동합니다.")
            print(f"STRATEGY: HARD \nWeighted1: None \nWeighted2: None \n**********")
            strategy_title = 'H'
            result = en.hard()
        elif args.WEIGHT1 and args.WEIGHT2:
            print(f"********** \nSTRATEGY: {args.STRATEGY} \nWeighted1: {args.WEIGHT1[0]} \nWeighted2: {args.WEIGHT2} \n**********")
            strategy_title = 'W1-'+'-'.join(map(str,*args.WEIGHT1)) + '_W2-' + str(args.WEIGHT2)
            result = en.weighted(*args.WEIGHT1, args.WEIGHT2)
        elif not args.WEIGHT1 and args.WEIGHT2:
            print(f"********** \nSTRATEGY: {args.STRATEGY} \nWeighted1: None \nWeighted2: {args.WEIGHT2} \n**********")
            strategy_title = 'W2-' + str(args.WEIGHT2)
            result = en.weighted([1] * en.csv_nums, args.WEIGHT2)
        else:
            print("잘못된 WEIGHT를 입력했습니다. 다시 실행해주세요.")
    else:
        raise KeyError("잘못된 STRATEGY을 선택했습니다. 다시 실행해주세요.")
    
    now = datetime.now(timezone("Asia/Seoul")).strftime(f"%Y%m%d-%H%M")
    result.to_csv(f'{savepath}{now}_{strategy_title}.csv',index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("--FILE_PATH", type=str,required=False,
        default="./ensembles_inference/",
        help='required: 앙상블 하고 싶은 csv 파일들이 있는 폴더의 경로를 입력해주세요.')
    arg('--STRATEGY', type=str, default='HARD',
        choices=['HARD', 'WEIGHTED'],
        help='optional: [HARD, WEIGHTED] 중 앙상블 전략을 선택해 주세요. (default="HARD")')
    arg('--WEIGHT1', nargs='+',default=None,
        type=lambda s: [float(item) for item in s.split(',')],
        help='optional: csv 사이의 가중치를 조정할 수 있습니다. 입력하지 않으면 Hard Voting과 같습니다.')
    arg('--WEIGHT2', type=float,default=0,
        help='optional: 순위 가중치 decay를 조정할 수 있습니다. 입력하지 않으면 WEIGHT1만 작동합니다.')
    arg('--RESULT_PATH',type=str, default='./ensembles_submit/',
        help='optional: 앙상블 결과를 저장할 폴더 경로를 전달합니다. (default:"./ensembles_submit/")')
    args = parser.parse_args()

    main(args)