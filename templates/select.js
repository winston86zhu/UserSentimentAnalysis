$("#model").change(function () {
    $("#trainer").toggle((+$(this).val()) === "bayes");
})