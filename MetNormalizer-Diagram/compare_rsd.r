compare_rsd <- function (sample.rsd, sample.nor.rsd, QC.rsd, QC.nor.rsd, path = ".") 
{
    temp_data1 <- data.frame(raw = sample.rsd, normalization = sample.nor.rsd, 
        stringsAsFactors = FALSE) %>% dplyr::mutate(class = dplyr::case_when(raw/normalization > 
        1 ~ "Decraese", raw/normalization == 1 ~ "Equal", raw/normalization < 
        1 ~ "Increase"))
    temp_data2 <- data.frame(raw = QC.rsd, normalization = QC.nor.rsd, 
        stringsAsFactors = FALSE) %>% dplyr::mutate(class = dplyr::case_when(raw/normalization > 
        1 ~ "Decraese", raw/normalization == 1 ~ "Equal", raw/normalization < 
        1 ~ "Increase"))
    plot1 <- ggplot2::ggplot(temp_data1) + ggplot2::geom_point(ggplot2::aes(x = raw, 
        y = normalization, colour = class)) + ggsci::scale_colour_aaas() + 
        ggplot2::labs(x = "Before normalization", y = "After normalization", 
            subtitle = "Subject samples") + ggplot2::scale_x_continuous(limits = c(0, 
        200)) + ggplot2::scale_y_continuous(limits = c(0, 200)) + 
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2, 
            colour = "black") + ggplot2::theme_bw()
    plot2 <- ggplot2::ggplot(temp_data2) + ggplot2::geom_point(ggplot2::aes(x = raw, 
        y = normalization, colour = class)) + ggsci::scale_colour_aaas() + 
        ggplot2::labs(x = "Before normalization", y = "After normalization", 
            subtitle = "QC samples") + ggplot2::scale_x_continuous(limits = c(0, 
        200)) + ggplot2::scale_y_continuous(limits = c(0, 200)) + 
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2, 
            colour = "black") + ggplot2::theme_bw()
    plot <- plot1 + plot2 + patchwork::plot_layout(nrow = 1)
    ggplot2::ggsave(plot, filename = file.path(path, "RSD compare plot.pdf"), 
        width = 14, height = 7)
}