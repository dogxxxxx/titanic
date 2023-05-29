import utils
import preprocess
import train
import predict


def main():
    train_df = utils.load_data(config["train_path"])
    test_df = utils.load_data(config["test_path"])

    x_train, y_train, x_val, y_val, x_test = preprocess.DNN_preprocess(
        train_df=train_df, test_df=test_df, config=config
    )
    # train.DNN_train(
    #     x_train=x_train,
    #     y_train=y_train,
    #     x_val=x_val,
    #     y_val=y_val,
    #     config=config,
    #     logger=logger,
    # )
    DNN_prediction = predict.DNN_predict(test_tensor=x_test, config=config)

    x_train, y_train, x_test = preprocess.rf_preprocess(
        train_df=train_df, test_df=test_df, config=config
    )
    # train.rf_train(x_train=x_train, y_train=y_train, config=config, logger=logger)
    rf_prediction = predict.rf_predict(
        test=x_test, model_path=config["rf_model_save_path"]
    )

    x_train, y_train, x_test = preprocess.xgboost_preprocess(
        train_df=train_df, test_df=test_df, config=config
    )
    # train.xgb_train(x_train=x_train, y_train=y_train, config=config, logger=logger)
    xgb_prediction = predict.xgb_predict(
        test_df=x_test, model_path=config["xgb_model_save_path"]
    )

    bagging_prediction = predict.bagging(DNN_prediction, rf_prediction, xgb_prediction)

    # utils.submit(
    #     prediction=DNN_prediction,
    #     test_df=test_df,
    #     submit_path=config["DNN_submit_path"],
    # )
    # utils.submit(
    #     prediction=rf_prediction, test_df=test_df, submit_path=config["rf_submit_path"]
    # )
    # utils.submit(xgb_prediction, test_df=test_df, submit_path=config["xgb_submit_path"])
    utils.submit(
        prediction=bagging_prediction,
        test_df=test_df,
        submit_path=config["bagging_submit_path"],
    )


if __name__ == "__main__":
    config = utils.load_config("config.yaml")
    logger = utils.set_logging(config["log_path"])
    main()
