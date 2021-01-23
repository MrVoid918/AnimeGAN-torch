from torchvision import transforms

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                # transforms.Normalize((red_mean, green_mean, blue_mean),
                                #                    (red_std, green_std, blue_std))
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

inv_gray_transform = transforms.Compose([transforms.Normalize([-1, -1, -1], [2., 2., 2.]),
                                         transforms.Grayscale(num_output_channels=3)])
