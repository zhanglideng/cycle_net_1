import xlwt


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def write_excel_train(sheet, line, epoch, itr, loss, weight):
    sum_loss = 0
    sheet.write(line, 0, epoch + 1)
    sheet.write(line, 1, itr + 1)
    for i in range(len(loss)):
        sheet.write(line, i + 2, round(loss[i], 6))
        sum_loss += loss[i] * weight[i]
    sheet.write(line, 2 + len(loss), round(sum_loss, 6))
    return line + 1


def write_excel_val(sheet, line, epoch, loss):
    loss, val, train = loss
    sheet.write(line, 0, epoch + 1)
    for i in range(len(loss)):
        sheet.write(line, i + 1, round(loss[i], 6))
    sheet.write(line, 1 + len(loss), round(val, 6))
    sheet.write(line, 2 + len(loss), round(train, 6))
    return line + 1


def write_excel_test(sheet, line, name, loss):
    # 0_a=0.86_b=1.01
    num = int(name[:4])
    if len(name)==4:
        air_light = 0.0
        beta = 0.0
    else:
        air_light = float(name[-11:-7])
        beta = float(name[-4:])
    sheet.write(line, 0, num)
    sheet.write(line, 1, air_light)
    sheet.write(line, 2, beta)
    for i in range(len(loss)):
        sheet.write(line, i + 3, round(loss[i], 6))
    return line + 1


def init_excel(kind):
    workbook = xlwt.Workbook()
    if kind == 'train':
        sheet1 = workbook.add_sheet('train', cell_overwrite_ok=True)
        sheet2 = workbook.add_sheet('val', cell_overwrite_ok=True)
        # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
        row0 = ["epoch", "itr",
                "J1_l2", "J1_ssim", "J1_vgg",
                "J2_l2", "J2_ssim", "J2_vgg",
                "J3_l2", "J3_ssim", "J3_vgg",
                "loss"]
        # row0 = ["epoch", "itr", "l2", "ssim", "loss"]
        row1 = ["epoch",
                "J1_l2", "J1_ssim", "J1_vgg",
                "J2_l2", "J2_ssim", "J2_vgg",
                "J3_l2", "J3_ssim", "J3_vgg",
                "val_loss", "train_loss"]
        # row1 = ["epoch", "l2", "ssim", "val_loss", "train_loss"]
        for i in range(0, len(row0)):
            print('写入train_excel')
            sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        for i in range(0, len(row1)):
            print('写入val_excel')
            sheet2.write(0, i, row1[i], set_style('Times New Roman', 220, True))
        return workbook, sheet1, sheet2
    elif kind == 'test':
        sheet1 = workbook.add_sheet('test', cell_overwrite_ok=True)
        # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
        row0 = ["num", "A", "beta",
                "J1_l2", "J1_ssim", "J1_vgg",
                "J2_l2", "J2_ssim", "J2_vgg",
                "J3_l2", "J3_ssim", "J3_vgg",
                "J4_l2", "J4_ssim", "J4_vgg",
                "J5_l2", "J5_ssim", "J5_vgg"]
        for i in range(0, len(row0)):
            print('写入test_excel')
            sheet1.write(0, i, row0[i], set_style('Times New Roman', 220, True))
        return workbook, sheet1
