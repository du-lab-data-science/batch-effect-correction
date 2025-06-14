peak_plot <- function (index, sample, sample.nor, QC, QC.nor, sample.order, 
    QC.order, tags, path = ".", sample.rsd = sample.rsd, QC.rsd = QC.rsd, 
    sample.nor.rsd = sample.nor.rsd, QC.nor.rsd = QC.nor.rsd) 
{
    for (i in index) {
        cat(i, " ")
        temp_data <- rbind(data.frame(order = sample.order, raw = sample[, 
            i], normalization = sample.nor[, 1], class = "Subject", 
            stringsAsFactors = FALSE), data.frame(order = QC.order, 
            raw = QC[, i], normalization = QC.nor[, 1], class = "QC", 
            stringsAsFactors = FALSE))
        plot_raw <- ggplot2::ggplot(temp_data, ggplot2::aes(x = order, 
            y = raw)) + ggplot2::geom_point(ggplot2::aes(colour = class)) + 
            ggplot2::labs(x = "Inction order", y = "Intensity", 
                title = paste("Peak:", tags["name", i]), subtitle = "Before") + 
            ggsci::scale_colour_lancet() + ggplot2::geom_smooth(ggplot2::aes(colour = class)) + 
            ggplot2::annotate(geom = "text", x = Inf, y = Inf, 
                hjust = 1, vjust = 1, label = paste("QC RSD:", 
                  round(QC.rsd[i], 2), "%\n", "Sample RSD:", 
                  round(sample.rsd[i], 2), "%")) + ggplot2::theme_bw()
        plot_normalization <- ggplot2::ggplot(temp_data, ggplot2::aes(x = order, 
            y = normalization)) + ggplot2::geom_point(ggplot2::aes(colour = class)) + 
            ggplot2::labs(x = "Inction order", y = "Intensity", 
                title = paste("Peak:", tags["name", i]), subtitle = "After") + 
            ggsci::scale_colour_lancet() + ggplot2::geom_smooth(ggplot2::aes(colour = class)) + 
            ggplot2::annotate(geom = "text", x = Inf, y = Inf, 
                hjust = 1, vjust = 1, label = paste("QC RSD:", 
                  round(QC.nor.rsd[i], 2), "%\n", "Sample RSD:", 
                  round(sample.nor.rsd[i], 2), "%")) + ggplot2::theme_bw()
        suppressMessages(require(patchwork))
        plot <- plot_raw + plot_normalization + patchwork::plot_layout(nrow = 2, 
            byrow = FALSE)
        ggplot2::ggsave(plot, filename = file.path(path, paste(tags["name", 
            i], "png", sep = ".")), width = 10, height = 7)
    }
}